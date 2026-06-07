import qlib

import os
_provider = os.environ.get("QLIB_DATA_DIR", os.environ.get("QLIB_PROVIDER_URI", "~/.qlib/qlib_data/cn_data"))
qlib.init(provider_uri=_provider)
from qlib.data import D

if __name__ == '__main__':
    instruments = D.instruments()
    fields = ["$open", "$close", "$high", "$low", "$volume"]  # , "$amount", "$turn", "$pettm", "$pbmrq"

    # Reduce date range to speed up: 2015-01-01 to now is too large (~10 years)
    # Use last 2 years only for MVP
    print("Loading data from 2023-01-01 to now...")
    data = D.features(instruments, fields, freq="day", start_time="2023-01-01").swaplevel().sort_index()

    # Calculate return
    data["$return"] = data.groupby(level=0)["$close"].pct_change().fillna(0)

    print(f"Loaded {len(data)} rows")
    print(data.head())

    data.to_hdf("./daily_pv_all.h5", key="data")
    print("✓ Saved to daily_pv_all.h5")

    # Debug version with first 100 instruments
    fields = ["$open", "$close", "$high", "$low", "$volume"]  # , "$amount", "$turn", "$pettm", "$pbmrq"
    data_debug = (
        (
            D.features(instruments, fields, freq="day", start_time="2023-01-01")
            .swaplevel()
            .sort_index()
        )
        .swaplevel()
        .loc[data.reset_index()["instrument"].unique()[:100]]
        .swaplevel()
        .sort_index()
    )

    # Calculate return
    data_debug["$return"] = data_debug.groupby(level=0)["$close"].pct_change().fillna(0)
    print(f"Debug data: {len(data_debug)} rows")
    data_debug.to_hdf("./daily_pv_debug.h5", key="data")
    print("✓ Saved to daily_pv_debug.h5")