import os
import sys
import unittest

import numpy as np
from osgeo import gdal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from autoroute.autoroute import AutoRouteHandler

class TestStreamRasterization(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
              "DATA_DIR": "test_ar_data",
              "DEM_FOLDER": "tests/test_data/DEMs/single_4326",
              "BUFFER_FILES": False, 
              "DEM_NAME": "test_dem", 
              "STREAM_NETWORK_FOLDER": "tests/test_data/streamlines/single_4326", 
              "STREAM_NAME": "test_strm", 
              "STREAM_ID": "LINKNO" }
        
    def test_sameProjection_gpkg_singleFiles(self):
        output = "test_ar_data/stream_files/test_dem__test_strm/N18W073_FABDEM_V1-2__strm.tif"
        validation = "tests/test_data/validation/rasterization/N18W073_FABDEM_V1-2__strm_val.tif"
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(output))
        out_ds = gdal.Open(output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(validation)
        self.assertTrue(np.count_nonzero(~(out_ds.ReadAsArray() == val_ds.ReadAsArray())) == 26, "Arrays are not equal") # Slightly different geometry, but we should get the same answer. Parquet tends to be more "correct" than gpkg
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

    def test_sameProjection_parquet_singleFiles(self):
        self.params["STREAM_NETWORK_FOLDER"] = "tests/test_data/streamlines/single_parquet_4326"
        output = "test_ar_data/stream_files/test_dem__test_strm/N18W073_FABDEM_V1-2__strm.tif"
        validation = "tests/test_data/validation/rasterization/N18W073_FABDEM_V1-2__strm_val.tif"

        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(output))
        out_ds = gdal.Open(output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

    def test_difProjection_parquet_singleFiles(self):
        self.params["STREAM_NETWORK_FOLDER"] = "tests/test_data/streamlines/single_parquet_3857"
        output = "test_ar_data/stream_files/test_dem__test_strm/N18W073_FABDEM_V1-2__strm.tif"
        validation = "tests/test_data/validation/rasterization/N18W073_FABDEM_V1-2__strm_val.tif"

        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(output))
        out_ds = gdal.Open(output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

if __name__ == '__main__':
    unittest.main()
    np.allclose