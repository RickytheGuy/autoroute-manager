import os
import glob
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

AUTOROUTE_EXE = os.path.join("tests","test_data","exes","AutoRoute_w_GDAL.exe")
FLOODSPREADER_EXE = os.path.join("tests","test_data","exes","AutoRoute_FloodSpreader.exe")

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

run_floodspreader = True
@unittest.skip
@unittest.skipIf(sys.platform != "win32" and sys.platform != "linux", "Runs only on windows or linux")
@unittest.skipIf(not os.path.exists(AUTOROUTE_EXE), "AutoRoute exe not found")
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
              "AUTOROUTE": AUTOROUTE_EXE,
               "AUTOROUTE_CONDA_ENV": "autoroute",
                "MANNINGS_TABLE": os.path.join("tests","test_data","mannings_table","mannings.txt") }
        self.output = os.path.join("test_ar_data","vdts","test_dem__test_strm","N18W073_FABDEM_V1-2__vdt.txt")
        self.validation = os.path.join("tests","test_data","validation","vdts","simple_inputs.txt")
        
    def tearDown(self) -> None:
        if self.output and os.path.exists(self.output): os.remove(self.output) 

    def test_AutoRoute(self):
        global run_floodspreader
        run_floodspreader = False
        AutoRouteHandler(self.params).run()
        2
        self.assertTrue(os.path.exists(self.output), f"File does not exist: {self.output}")
        with open(self.output) as f:
            out = f.read()
        with open(self.validation) as f:
            val = f.read()
        self.assertEqual(out, val, "VDT files are not equal")
        run_floodspreader = True

    def test_AutoRoute_lots_of_params(self):
        self.params["RAPID_Subtract_BaseFlow"] = True
        self.params["VDT"] = "test_ar_data"
        self.params["num_iterations"] = 5
        self.params["convert_cfs_to_cms"] = True
        self.params["x_distance"] = 100
        self.params["q_limit"] = 1.01
        self.params["direction_distance"] = 10
        self.params["slope_distance"] = 10
        self.params["weight_angles"] = 1
        self.params["use_prev_d_4_xs"] = 0
        self.params["adjust_flow"] = 0.9
        self.params["row_start"] = 0
        self.params["row_end"] = 1000
        self.params["degree_manip"] = 6.1
        self.params["degree_interval"] = 1.5
        self.params["man_n"] = 0.01
        self.params["low_spot_distance"] = 30
        self.params["low_spot_is_meters"] = True
        self.params["low_spot_use_box"] = True
        self.params["box_size"] = 5
        self.params["find_flat"] = True
        self.params["ar_bathy_file"] = 'test_ar_data'
        self.params["bathy_alpha"] = 0.001
        self.params["bathy_method"] = 'Trapezoidal'
        self.params["bathy_x_max_depth"] = 0.3
        self.params["bathy_y_shallow"] =  0.3
        self.output = os.path.join("test_ar_data","N18W073_FABDEM_V1-2__ar_bathy.tif")
        self.validation = os.path.join("tests","test_data","validation","bathy","ar_bathy_all_params.tif")

        global run_floodspreader
        run_floodspreader = False
        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output), f"File does not exist: {self.output}")
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        self.assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
        self.assertTrue(np.isclose((np.array(out_ds.GetGeoTransform()) - np.array(val_ds.GetGeoTransform())).max(), 0), "GeoTransform is not equal")
        self.assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

        out_ds = None
        os.remove(self.output)

        self.output = os.path.join("test_ar_data","N18W073_FABDEM_V1-2__vdt.txt")
        self.validation = os.path.join("tests","test_data","validation","vdts","all_params.txt")
        
        self.assertTrue(os.path.exists(self.output), f"File does not exist: {self.output}")
        with open(self.output) as f:
            out = f.read()
        with open(self.validation) as f:
            val = f.read()
        self.assertEqual(out, val, "VDT files are not equal")
        run_floodspreader = True

#@unittest.skip
@unittest.skipIf(sys.platform != "win32" and sys.platform != "linux", "Runs only on windows or linux")
@unittest.skipIf(not os.path.exists(os.path.join(AUTOROUTE_EXE)) or not os.path.exists(FLOODSPREADER_EXE), "AutoRoute and/or FloodSpreader not found")
@unittest.skipIf(not run_floodspreader, "Autoroute failed, not running FloodSpreader")
class TestFloodSpreader(unittest.TestCase):
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
              "FLOOD_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flowfile.csv",),
              "AUTOROUTE": AUTOROUTE_EXE,
              "FLOODSPREADER": FLOODSPREADER_EXE,
              "AUTOROUTE_CONDA_ENV": "autoroute",
              "MANNINGS_TABLE": os.path.join("tests","test_data","mannings_table","mannings.txt"),
               "FLOOD_MAP":os.path.join("test_ar_data","simple_map") }
        self.output = os.path.join("test_ar_data","simple_map","N18W073_FABDEM_V1-2__flood.tif")
        self.validation = os.path.join("tests","test_data","validation","maps","simple_flood.tif")
        self.tearDown()
        
    def tearDown(self) -> None:
        try:
            {os.remove(m) for m in glob.glob(os.path.join("test_ar_data","*bathy.tif"))}
            if self.output and os.path.exists(self.output): os.remove(self.output) 
        except PermissionError:
            print(f"\nCould not remove {self.output}\n")

    def test_FloodSpreader(self):
        AutoRouteHandler(self.params).run()
        
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
        out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
        out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
        out_ds = None
        self.assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
        self.assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
        self.assertEqual(out_proj, val_proj, "Projection is not equal")
        

    def test_FloodSpreader_lots_of_params(self):
        self.params["ar_bathy_file"] = 'test_ar_data'
        self.params["bathy_alpha"] = 0.001
        self.params["bathy_method"] = 'Trapezoidal'

        self.params["omit_outliers"] = 'Smooth Water Surface Elevation'
        self.params["wse_search_dist"] = 15
        self.params["wse_threshold"] = 0.2
        self.params["wse_remove_three"] = True
        self.params["specify_depth"] = 0
        self.params["twd_factor"] = 2
        self.params["only_streams"] = False
        self.params["use_ar_top_widths"] = True
        self.params["flood_local"] = True
        self.params["fs_bathy_file"] = os.path.join("test_ar_data","many_bathy")
        self.params["fs_bathy_smooth_method"] = ''
        self.params["bathy_twd_factor"] = 1

        self.output = os.path.join("test_ar_data","many_bathy","N18W073_FABDEM_V1-2__fs_bathy.tif")
        self.validation = os.path.join("tests","test_data","validation","bathy","fs_bathy_all_params.tif")

        AutoRouteHandler(self.params).run()

        self.assertTrue(os.path.exists(self.output), f"File does not exist: {self.output}")
        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
        out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
        out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
        out_ds = None
        val_ds = None
        self.assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
        self.assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
        self.assertEqual(out_proj, val_proj, "Projection is not equal")

        os.remove(self.output)

        self.output = os.path.join("test_ar_data","simple_map","N18W073_FABDEM_V1-2__flood.tif")
        self.validation = os.path.join("tests","test_data","validation","maps","all_params_flood.tif")

        out_ds = gdal.Open(self.output)
        self.assertIsNotNone(out_ds, "Problem opening file")
        val_ds = gdal.Open(self.validation)
        out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
        out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
        out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
        out_ds = None
        val_ds = None
        self.assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
        self.assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
        self.assertEqual(out_proj, val_proj, "Projection is not equal")


if __name__ == '__main__':
    unittest.main()