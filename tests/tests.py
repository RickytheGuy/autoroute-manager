import os
import sys
import unittest
import logging

import numpy as np
import pandas as pd
from osgeo import gdal

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from autoroute.autoroute import AutoRouteHandler

# logging.basicConfig(level=logging.INFO,
#                     stream=sys.stdout,
#                     format='%(asctime)s - %(levelname)s - %(message)s')

run_extent=True
@unittest.skip
class TestStreamRasterization(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
              "DATA_DIR": "test_ar_data",
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "DEM_NAME": "test_dem", 
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_4326"), 
              "STREAM_NAME": "test_strm", 
              "STREAM_ID": "LINKNO" }
        self.output = os.path.join("test_ar_data","stream_files","test_dem__test_strm","N18W073_FABDEM_V1-2__strm.tif")
        self.validation = os.path.join("tests","test_data","validation","rasterization","N18W073_FABDEM_V1-2__strm_val.tif")
        
        
    def tearDown(self) -> None:
        if self.output and os.path.exists(self.output): os.remove(self.output) 
        
    def test_sameProjection_gpkg_singleFiles(self):
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.count_nonzero(~(out_ds.ReadAsArray() == val_ds.ReadAsArray())) <= 26, "Arrays are not equal") # Slightly different geometry, but we should get the same answer. Parquet tends to be more "correct" than gpkg
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")
        out_ds = None
        val_ds = None

    def test_sameProjection_parquet_singleFiles(self):
        self.params["STREAM_NETWORK_FOLDER"] = os.path.join("tests","test_data","streamlines","single_parquet_4326")
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")
        out_ds = None
        val_ds = None

    def test_difProjection_parquet_singleFiles(self):
        global run_extent
        run_extent=False
        self.params["STREAM_NETWORK_FOLDER"] = os.path.join("tests","test_data","streamlines","single_parquet_3857")
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")
        run_extent=True
        out_ds = None
        val_ds = None

    def test_various_files_and_projections(self):
        self.params["STREAM_NETWORK_FOLDER"] = os.path.join("tests","test_data","streamlines","multiple_parquet_various")

        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.count_nonzero(~(out_ds.ReadAsArray() == val_ds.ReadAsArray())) <= 17, "Arrays are not equal") # Slightly different, due to gpkg again
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")
        out_ds = None
        val_ds = None

    @unittest.skipIf(not run_extent, "one of the parquet tests failed, not running this one")
    def test_w_extent(self):
        self.params["STREAM_NETWORK_FOLDER"] = os.path.join("tests","test_data","streamlines","single_parquet_4326")
        self.params["EXTENT"] = (-72.1626, 18.6228, -72.1195, 18.6611)
        self.params["CROP"] = True
        self.output = os.path.join("test_ar_data","stream_files","test_dem__test_strm","-72_163__18_623__-72_12__18_661_crop__strm.tif")
        self.validation = os.path.join("tests","test_data","validation","rasterization","-72_163__18_623__-72_12__18_661_crop__strm.tif")
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds)
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")
        out_ds = None
        val_ds = None

@unittest.skip
class TestRowColIdFIle(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
              "DATA_DIR": "test_ar_data",
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "DEM_NAME": "test_dem", 
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
              "STREAM_NAME": "test_strm", 
              "STREAM_ID": "LINKNO",
               "SIMULATION_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flows.csv",),
               "ID_COLUMN": "LINKNO",
               "FLOW_COLUMN": "max",
               "BASE_FLOW_COLUMN": "flow"}
        self.output = os.path.join("test_ar_data","rapid_files","test_dem__test_strm","N18W073_FABDEM_V1-2__strm__row_col_id.txt")
        self.validation = os.path.join("tests","test_data","validation","row_id_flow","N18W073_FABDEM_V1-2__strm__row_col_id.txt")
 
    def tearDown(self) -> None:
        if self.output and os.path.exists(self.output): os.remove(self.output) 

    def test_row_col_id_file(self):
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_df = pd.read_csv(self.output, sep='\t')
        val_df = pd.read_csv(self.validation, sep='\t')
        out_df = out_df.reindex(sorted(out_df.columns), axis=1)
        val_df = val_df.reindex(sorted(val_df.columns), axis=1)
        self.assertTrue(out_df.equals(val_df), "Dataframes are not equal")

    def test_row_col_id_file_no_inputs(self):
        self.params["ID_COLUMN"] = ""
        self.params["FLOW_COLUMN"] = ""
        self.params["BASE_FLOW_COLUMN"] = ""
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_df = pd.read_csv(self.output, sep='\t')
        val_df = pd.read_csv(self.validation, sep= '\t')
        out_df = out_df.reindex(sorted(out_df.columns), axis=1)
        val_df = val_df.reindex(sorted(val_df.columns), axis=1)
        self.assertTrue(out_df.equals(val_df), "Dataframes are not equal")

@unittest.skip
class TestLandUse(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
              "DATA_DIR": os.path.join("test_ar_data"),
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "DEM_NAME": "test_dem", 
              "LAND_USE_NAME": "test_land_use", }
        self.output = os.path.join("test_ar_data","land_use","test_dem__test_land_use","N18W073_FABDEM_V1-2__lu.vrt")
        self.validation = os.path.join("tests","test_data","validation","LU","same_proj","lu.tif")
        
    def tearDown(self) -> None:
        if not os.path.exists(self.output):
            self.output = self.output.replace(".vrt", ".tif")
        if self.output and os.path.exists(self.output): os.remove(self.output) 
        
    def test_land_use_samesize(self):
        self.params['LAND_USE_FOLDER'] = os.path.join("tests","test_data","LUs","single_4326")
        AutoRouteHandler(self.params).run()

        if not os.path.exists(self.output):
            self.output = self.output.replace(".vrt", ".tif")
        self.assertTrue(os.path.exists(self.output), "File does not exist")
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

    def test_land_use_multiple_projected(self):
        self.params['LAND_USE_FOLDER'] = os.path.join("tests","test_data","LUs","multiple_nad")
        self.validation = os.path.join("tests","test_data","validation","LU","dif_proj","lu.tif") # Projection slightly rotates output, which is close enough
        AutoRouteHandler(self.params).run()

        if not os.path.exists(self.output):
            self.output = self.output.replace(".vrt", ".tif")
        self.assertTrue(os.path.exists(self.output), "File does not exist")
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertEqual(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

@unittest.skip
class TestCrop(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
              "DATA_DIR": "test_ar_data",
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "DEM_NAME": "test_dem", 
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326", ),
              "STREAM_NAME": "test_strm", 
              "STREAM_ID": "LINKNO" }
        self.output = os.path.join("test_ar_data","dems","test_dem","-72_163__18_623__-72_12__18_661_crop.vrt")
        self.validation = os.path.join("tests","test_data","DEMs","single_cropped","cropped_dem.tif")
        
    def tearDown(self) -> None:
        if not os.path.exists(self.output):
            self.output = self.output.replace(".vrt", ".tif")
        if self.output and os.path.exists(self.output): os.remove(self.output) 
        
    def test_crop_dem(self):
        self.params["EXTENT"] = (-72.1626, 18.6228, -72.1195, 18.6611)
        self.params["CROP"] = True
        AutoRouteHandler(self.params).run()

        if not os.path.exists(self.output):
            self.output = self.output.replace(".vrt", ".tif")
        self.assertTrue(os.path.exists(self.output), "File does not exist")
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        self.assertTrue((out_ds.ReadAsArray() - val_ds.ReadAsArray()).max() < 9.2, "Arrays are not equal") # Because cells are shifted, values shift as well oh so slightly, which a vrt does not pick up
        self.assertTrue(np.isclose((np.array(out_ds.GetGeoTransform()) - np.array(val_ds.GetGeoTransform())).max(), 0), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

@unittest.skip
class TestFlowFile(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
                       "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
                       "DATA_DIR": "test_ar_data",
                       "DEM_NAME": "test_dem", 
                       "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
                       "STREAM_NAME": "test_strm", 
                       "STREAM_ID": "LINKNO",
                       "FLOOD_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flowfile.csv",),
               }
        self.output = os.path.join("test_ar_data","flow_files","test_dem__test_strm","N18W073_FABDEM_V1-2__strm__flow.txt")
        self.validation = os.path.join("tests","test_data","validation","flowfile","N18W073_FABDEM_V1-2__strm__flow.txt")
 
    def tearDown(self) -> None:
        if self.output and os.path.exists(self.output): os.remove(self.output) 

    def test_flowfile(self):
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output))
        out_df = pd.read_csv(self.output)
        val_df = pd.read_csv(self.validation)
        self.assertTrue(out_df.equals(val_df), "Dataframes are not equal")

@unittest.skipIf(sys.platform != "win32" and sys.platform != "linux", "Runs only on windows or linux")
@unittest.skipIf(not os.path.exists(os.path.join("tests","test_data","exes","AutoRoute_w_GDAL.exe")), "AutoRoute exe not found")
class TestAutoRoute(unittest.TestCase):
    def setUp(self) -> None:
        self.params = {"OVERWRITE": True,
              "DATA_DIR": "test_ar_data",
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "DEM_NAME": "test_dem", 
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"), 
              "STREAM_NAME": "test_strm", 
              "STREAM_ID": "LINKNO",
              "LAND_USE_FOLDER": os.path.join("tests","test_data","LUs","single_4326"),
              "LAND_USE_NAME": "test_land",
              "SIMULATION_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flows.csv",),
               "SIMULATION_ID_COLUMN": "LINKNO",
               "SIMULATION_FLOW_COLUMN": "max",
               "BASE_FLOW_COLUMN": "flow",
              "AUTOROUTE": os.path.join("tests","test_data","exes","AutoRoute_w_GDAL.exe"),
              "FLOODSPREADER": os.path.join("tests","test_data","exes", "AutoRoute_FloodSpreader.exe"),
               "AUTOROUTE_CONDA_ENV": "autoroute",
                "MANNINGS_TABLE": os.path.join("tests","test_data","mannings_table","mannings.txt") }
        self.output = os.path.join("test_ar_data","vdts","test_dem__test_strm","N18W073_FABDEM_V1-2__vdt.txt")
        self.validation = os.path.join("tests","test_data","validation","vdts","simple_inputs.txt")
        
    def tearDown(self) -> None:
        if self.output and os.path.exists(self.output): os.remove(self.output) 

    def test_AutoRoute(self):
        AutoRouteHandler(self.params).run()
        2
        self.assertTrue(os.path.exists(self.output), f"File does not exist: {self.output}")
        with open(self.output) as f:
            out = f.read()
        with open(self.validation) as f:
            val = f.read()
        self.assertEqual(out, val, "VDT files are not equal")

if __name__ == '__main__':
    unittest.main()