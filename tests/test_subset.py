from pathlib import Path

import pytest
import xarray as xr
from numpy.testing import assert_array_equal

import grib2io

# List of all GRIB2 files in test data
GRIB2_FILES = [
    "2024101012_Milton_Adv22_e70_cum_dat.grb",
    "blend.t00z.core.f001.co_4x_reduce.grib2",
    "ds.temp.bin",
    "gefs.chem.t00z.a2d_0p25.f000.grib2_subset",
    "gfs.complex.grib2",
    "gfs.jpeg.grib2",
    "gfs.png.grib2",
    "gfs.t00z.pgrb2.1p00.f024",
    "gfs_20221107/gfs.t00z.pgrb2.1p00.f009_subset",
    # "gfs_20221107/gfs.t00z.pgrb2.1p00.f012_subset",
    # "gfs_20221107/gfs.t06z.pgrb2.1p00.f009_subset",
    # "gfs_20221107/gfs.t06z.pgrb2.1p00.f012_subset",
    # "rap.tsoil.gdt32769.grib2",  # Grid not currently supported by subset.
]


parametrize_data = [
    pytest.param(
        "rap.tsoil.gdt32769.grib2",
        (43, 32.7),
        (-281, -243),
        ValueError,
        id="RAP grid not supported.",
    ),
    pytest.param(
        "gfs_20221107/gfs.t00z.pgrb2.1p00.f012_subset",
        (43, 32.7),
        (-243, -281),
        ValueError,
        id="Invalid longitude order.",
    ),
    pytest.param(
        "gfs_20221107/gfs.t00z.pgrb2.1p00.f012_subset",
        (32.7, 43),
        (-281, -243),
        ValueError,
        id="Invalid latitude order.",
    ),
    pytest.param(
        "gfs_20221107/gfs.t00z.pgrb2.1p00.f012_subset",
        None,
        (-281, -243),
        [
            0,
            429,
            0,
            0,
            0,
            6,
            0,
            0,
            0,
            0,
            0,
            0,
            39,
            11,
            0,
            -1,
            43000000,
            79000000,
            48,
            33000000,
            117000000,
            1000000,
            1000000,
            0,
        ],
        id="Valid subset with lats=None.",
    ),
    pytest.param(
        "gfs_20221107/gfs.t00z.pgrb2.1p00.f009_subset",
        (32.7, 43),
        None,
        [
            0,
            429,
            0,
            0,
            0,
            6,
            0,
            0,
            0,
            0,
            0,
            0,
            39,
            11,
            0,
            -1,
            43000000,
            79000000,
            48,
            33000000,
            117000000,
            1000000,
            1000000,
            0,
        ],
        id="Valid subset with lons=None.",
    ),
    pytest.param(
        "gfs_20221107/gfs.t00z.pgrb2.1p00.f012_subset",
        None,
        None,
        [
            0,
            429,
            0,
            0,
            0,
            6,
            0,
            0,
            0,
            0,
            0,
            0,
            39,
            11,
            0,
            -1,
            43000000,
            79000000,
            48,
            33000000,
            117000000,
            1000000,
            1000000,
        ],
        id="Valid subset with lats=None and lons=None",
    ),
    pytest.param(
        "gfs_20221107/gfs.t00z.pgrb2.1p00.f012_subset",
        (32.7, 43),
        (-281, -243),
        [
            0,
            429,
            0,
            0,
            0,
            6,
            0,
            0,
            0,
            0,
            0,
            0,
            39,
            11,
            0,
            -1,
            43000000,
            79000000,
            48,
            33000000,
            117000000,
            1000000,
            1000000,
        ],
        id="Valid subset",
    ),
]


@pytest.mark.parametrize(
    "filename, lats, lons, expected",
    parametrize_data,
)
def test_message_subset(request, filename, lats, lons, expected):
    """Test subsetting messages from various GRIB2 files."""
    datadir = request.config.rootdir / "tests" / "input_data"
    filepath = datadir / filename

    with grib2io.open(filepath) as msgs:
        if isinstance(expected, type) and issubclass(expected, Exception):
            with pytest.raises(expected):
                msgs[0].subset(lats=lats, lons=lons)
        else:
            newmsg = msgs[0].subset(lats=lats, lons=lons)
            assert_array_equal(newmsg.section3, expected)


#@pytest.mark.parametrize(
#    "filename, lats, lons, expected",
#    parametrize_data,
#)
#def test_dataarray_subset(request, filename, lats, lons, expected):
#    """Test subsetting messages from various GRIB2 files."""
#    datadir = request.config.rootdir / "tests" / "input_data"
#    filepath = datadir / filename
#
#    ds = xr.open_dataset(
#        filepath,
#        engine="grib2io",
#    )
#    da = ds[0]
#    if isinstance(expected, type) and issubclass(expected, Exception):
#        with pytest.raises(expected):
#            da.subset(lats=lats, lons=lons)
#    else:
#        newmsg = da.subset(lats=lats, lons=lons)
#        assert_array_equal(newmsg.section3, expected)


@pytest.mark.parametrize("filename", GRIB2_FILES)
def test_file_open_and_basic_subset(request, filename):
    """Test that all GRIB2 files can be opened and basic subset operations work."""
    datadir = request.config.rootdir / "tests" / "input_data"
    filepath = datadir / filename

    with grib2io.open(filepath) as msgs:
        # Try a basic subset operation that should work for most files
        msg = msgs[0]
        # Get grid info to determine reasonable subset bounds
        lats, lons = msg.latlons()
        lat_min, lat_max = float(lats.min()), float(lats.max())
        lon_min, lon_max = float(lons.min()), float(lons.max())

        # Create a small subset in the middle of the grid
        lat_center = (lat_min + lat_max) / 2
        lon_center = (lon_min + lon_max) / 2
        lat_range = (lat_max - lat_min) * 0.1  # 10% of range
        lon_range = (lon_max - lon_min) * 0.1

        subset_lats = (lat_center - lat_range / 2, lat_center + lat_range / 2)
        subset_lons = (lon_center - lon_range / 2, lon_center + lon_range / 2)

        # Attempt subset - should not raise exception
        subset_msg = msg.subset(lats=subset_lats, lons=subset_lons)
        assert subset_msg is not None


def test_xarray_subset_original_case(request):
    """Test the original xarray subset functionality."""
    datadir = request.config.rootdir / "tests" / "input_data"

    filters = {
        "typeOfFirstFixedSurface": 103,
        "valueOfFirstFixedSurface": 2,
        "productDefinitionTemplateNumber": 0,
        "shortName": "TMP",
    }

    ds = xr.open_mfdataset(
        [
            datadir / "gfs_20221107" / "gfs.t00z.pgrb2.1p00.f009_subset",
            datadir / "gfs_20221107" / "gfs.t00z.pgrb2.1p00.f012_subset",
        ],
        combine="nested",
        concat_dim="leadTime",
        engine="grib2io",
        filters=filters,
        coords="different",
        compat="no_conflicts",
    )

    lats, lons = (32.7, 43), (79 - 360, 117 - 360)
    expected = [
        0,
        429,
        0,
        0,
        0,
        6,
        0,
        0,
        0,
        0,
        0,
        0,
        39,
        11,
        0,
        -1,
        43000000,
        79000000,
        48,
        33000000,
        117000000,
        1000000,
        1000000,
        0,
    ]

    newds_xr = ds["TMP"].grib2io.subset(lats=lats, lons=lons)
    newds_gr = ds.grib2io.subset(lats=lats, lons=lons)

    assert_array_equal(newds_xr.attrs["GRIB2IO_section3"], expected)
    assert_array_equal(newds_gr["TMP"].attrs["GRIB2IO_section3"], expected)


#@pytest.mark.parametrize("filename", GRIB2_FILES)
#def test_subset_none_parameters(request, filename):
#    """Test subset with None parameters."""
#    datadir = request.config.rootdir / "tests" / "input_data"
#    filepath = datadir / filename
#
#    with grib2io.open(filepath) as msgs:
#        msg = msgs[0]
#        lats, lons = msg.latlons()
#
#        # Get subset bounds
#        lat_min, lat_max = float(lats.min()), float(lats.max())
#        lon_min, lon_max = float(lons.min()), float(lons.max())
#        lat_center = (lat_min + lat_max) / 2
#        lon_center = (lon_min + lon_max) / 2
#        lat_range = (lat_max - lat_min) * 0.1
#        lon_range = (lon_max - lon_min) * 0.1
#
#        subset_lats = (lat_center - lat_range / 2, lat_center + lat_range / 2)
#        subset_lons = (lon_center - lon_range / 2, lon_center + lon_range / 2)
#
#        # Test lats=None, lons=subset_lons
#        result1 = msg.subset(lats=None, lons=subset_lons)
#        assert result1 is not None
#
#        # Test lats=subset_lats, lons=None
#        result2 = msg.subset(lats=subset_lats, lons=None)
#        assert result2 is not None
#
#        # Test lats=None, lons=None (should return original message)
#        result3 = msg.subset(lats=None, lons=None)
#        assert_array_equal(result3.section3, msg.section3)
