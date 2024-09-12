import os
import sys

import pytest
import numpy as np
import pandas as pd
from osgeo import gdal

from autoroute_manager.autoroute import AutoRoute

AUTOROUTE_EXE = os.path.join("tests","test_data","exes","AutoRoute_w_GDAL.exe")
FLOODSPREADER_EXE = os.path.join("tests","test_data","exes","AutoRoute_FloodSpreader.exe")

def run_autoroute(params, output):
    AutoRoute(params).run()
    assert os.path.exists(output)

def check_row_col_equals(params, output, validation):
    run_autoroute(params, output)

    out_df = pd.read_csv(output, sep='\t')
    val_df = pd.read_csv(validation, sep='\t')
    out_df = out_df.reindex(sorted(out_df.columns), axis=1)
    val_df = val_df.reindex(sorted(val_df.columns), axis=1)
    pd.testing.assert_frame_equal(out_df, val_df, check_exact=False)

def check_arrays_equal(output, validation):
    with gdal.Open(output) as out_ds, gdal.Open(validation) as val_ds:
        assert out_ds is not None
        assert np.allclose(out_ds.ReadAsArray(), val_ds.ReadAsArray())
        assert np.allclose(out_ds.GetGeoTransform(), val_ds.GetGeoTransform())
        assert out_ds.GetProjection() == val_ds.GetProjection()

if __name__ in ["__main__", 'test_autoroute']:
    def tearDown(output) -> None:
        if output and os.path.exists(output): os.remove(output) 

    ####################################################################################################
    #### Stream Rasterization Tests
    ####################################################################################################
    run_extent=True  
        
    def test_strm_rasterization():   
        params = {"OVERWRITE": True,
                "DATA_DIR": "test_ar_data",
                "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
                "BUFFER_FILES": False, 
                "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_4326"), 
                "STREAM_NAME": "test_strm", 
                "STREAM_ID": "LINKNO", 
                "DISABLE_PBAR": True}
        output = os.path.join("test_ar_data","stream_files","N18W073_FABDEM_V1-2__strm.tif")
        validation = os.path.join("tests","test_data","validation","rasterization","N18W073_FABDEM_V1-2__strm_val.tif")
        AutoRoute(params).run()

        assert os.path.exists(output)
        with gdal.Open(output) as out_ds, gdal.Open(validation) as val_ds:
            assert out_ds is not None
            assert np.count_nonzero(~(out_ds.ReadAsArray() == val_ds.ReadAsArray())) <= 26 # Slightly different geometry, but we should get the same answer. Parquet tends to be more "correct" than gpkg
            assert out_ds.GetGeoTransform() == val_ds.GetGeoTransform()
            assert out_ds.GetProjection() == val_ds.GetProjection()
        tearDown(output)
        
    # test_strm_rasterization()

    def test_sameProjection_parquet_singleFiles():
        params = {"OVERWRITE": True,
                "DATA_DIR": "test_ar_data",
                "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
                "BUFFER_FILES": False, 
                "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_4326"), 
                "STREAM_NAME": "test_strm", 
                "STREAM_ID": "LINKNO", 
                "DISABLE_PBAR": True,
                "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326")}
        output = os.path.join("test_ar_data","stream_files","N18W073_FABDEM_V1-2__strm.tif")
        validation = os.path.join("tests","test_data","validation","rasterization","N18W073_FABDEM_V1-2__strm_val.tif")
        AutoRoute(params).run()

        check_arrays_equal(output, validation)
        tearDown(output)

    def test_difProjection_parquet_singleFiles():
        global run_extent
        run_extent=False
        params = {"OVERWRITE": True,
            "DATA_DIR": "test_ar_data",
            "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
            "BUFFER_FILES": False, 
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_4326"), 
            "STREAM_NAME": "test_strm", 
            "STREAM_ID": "LINKNO", 
            "DISABLE_PBAR": True,
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_3857")
        }
        AutoRoute(params).run()
        
        output = os.path.join("test_ar_data","stream_files","N18W073_FABDEM_V1-2__strm.tif")
        validation = os.path.join("tests","test_data","validation","rasterization","N18W073_FABDEM_V1-2__strm_val.tif")

        check_arrays_equal(output, validation)
        run_extent=True
        tearDown(output)

    def test_various_files_and_projections():
        params = {"OVERWRITE": True,
            "DATA_DIR": "test_ar_data",
            "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
            "BUFFER_FILES": False, 
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_4326"), 
            "STREAM_NAME": "test_strm", 
            "STREAM_ID": "LINKNO", 
            "DISABLE_PBAR": True,
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","multiple_parquet_various")
        }

        AutoRoute(params).run()
        
        output = os.path.join("test_ar_data","stream_files","N18W073_FABDEM_V1-2__strm.tif")
        validation = os.path.join("tests","test_data","validation","rasterization","N18W073_FABDEM_V1-2__strm_val.tif")

        assert os.path.exists(output)
        with gdal.Open(output) as out_ds, gdal.Open(validation) as val_ds:
            assert out_ds is not None
            assert np.count_nonzero(~(out_ds.ReadAsArray() == val_ds.ReadAsArray())) <= 17 # Slightly different geometry, but we should get the same answer. Parquet tends to be more "correct" than gpkg
            assert out_ds.GetGeoTransform() == val_ds.GetGeoTransform()
            assert out_ds.GetProjection() == val_ds.GetProjection()
        tearDown(output)

    @pytest.mark.skipif(not run_extent, reason="one of the parquet tests failed, not running this one")
    def test_w_extent():
        params = {"OVERWRITE": True,
            "DATA_DIR": "test_ar_data",
            "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
            "BUFFER_FILES": False, 
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_4326"), 
            "STREAM_NAME": "test_strm", 
            "STREAM_ID": "LINKNO", 
            "DISABLE_PBAR": True,
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
            "EXTENT": (-72.1626, 18.6228, -72.1195, 18.6611),
            "CROP": True
        }
        output = os.path.join("test_ar_data","stream_files","N18W073_FABDEM_V1-2_crop__strm.tif")
        validation = os.path.join("tests","test_data","validation","rasterization","-72_163__18_623__-72_12__18_661_crop__strm.tif")
        AutoRoute(params).run()

        check_arrays_equal(output, validation)
        tearDown(output)

    #### RAPID File Creation Tests
    
    def test_row_col_id_file():
        params = {"OVERWRITE": True,
            "DATA_DIR": "test_ar_data",
            "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
            "DEM_NAME": "test_dem", 
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
            "STREAM_NAME": "test_strm", 
            "STREAM_ID": "LINKNO",
            "SIMULATION_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flows.csv",),
            "ID_COLUMN": "LINKNO",
            "FLOW_COLUMN": "max",
            "BASE_FLOW_COLUMN": "flow",
            "DISABLE_PBAR": True}
        output = os.path.join("test_ar_data","rapid_files","N18W073_FABDEM_V1-2__row_col_id.txt")
        validation = os.path.join("tests","test_data","validation","row_id_flow","N18W073_FABDEM_V1-2__strm__row_col_id.txt")
        
        check_row_col_equals(params, output, validation)
        tearDown(output)

    def test_row_col_id_file_no_inputs():
        params = {"OVERWRITE": True,
            "DATA_DIR": "test_ar_data",
            "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
            "DEM_NAME": "test_dem", 
            "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
            "STREAM_NAME": "test_strm", 
            "STREAM_ID": "LINKNO",
            "SIMULATION_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flows.csv",),
            "ID_COLUMN": "",
            "FLOW_COLUMN": "",
            "BASE_FLOW_COLUMN": "",
            "DISABLE_PBAR": True}
        
        output = os.path.join("test_ar_data","rapid_files","N18W073_FABDEM_V1-2__row_col_id.txt")
        validation = os.path.join("tests","test_data","validation","row_id_flow","N18W073_FABDEM_V1-2__strm__row_col_id.txt")

        check_row_col_equals(params, output, validation)
        tearDown(output)
        
    #### Land Use Tests
        
    def test_land_use_samesize():
        params = {"OVERWRITE": True,
              "DATA_DIR": os.path.join("test_ar_data"),
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "STREAM_ID": "LINKNO",
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
               "DISABLE_PBAR": True,
               'LAND_USE_FOLDER': os.path.join("tests","test_data","LUs","single_4326")
        }
        output = os.path.join("test_ar_data","land_use","N18W073_FABDEM_V1-2__lu.vrt")
        validation = os.path.join("tests","test_data","validation","LU","same_proj","lu.tif")
        AutoRoute(params).run()

        if not os.path.exists(output):
            output = output.replace(".vrt", ".tif")
        check_arrays_equal(output, validation)
        tearDown(output)

    def test_land_use_multiple_projected():
        params = {"OVERWRITE": True,
              "DATA_DIR": os.path.join("test_ar_data"),
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "STREAM_ID": "LINKNO",
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
               "DISABLE_PBAR": True,
               'LAND_USE_FOLDER': os.path.join("tests","test_data","LUs","multiple_nad")
        }
        output = os.path.join("test_ar_data","land_use","N18W073_FABDEM_V1-2__lu.vrt")
        validation = os.path.join("tests","test_data","validation","LU","dif_proj","lu.tif") # Projection slightly rotates output, which is close enough
        AutoRoute(params).run()

        if not os.path.exists(output):
            output = output.replace(".vrt", ".tif")
        check_arrays_equal(output, validation)
        tearDown(output)

    def test_crop_dem():
        params = {"OVERWRITE": True,
              "DATA_DIR": "test_ar_data",
              "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
              "BUFFER_FILES": False, 
              "DEM_NAME": "test_dem", 
              "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326", ),
              "STREAM_NAME": "test_strm", 
              "STREAM_ID": "LINKNO",
               "DISABLE_PBAR": True}
        output = os.path.join("test_ar_data","dems","cropped","N18W073_FABDEM_V1-2_crop.vrt")
        validation = os.path.join("tests","test_data","DEMs","single_cropped","cropped_dem.tif")
        
        params["EXTENT"] = (-72.1626, 18.6228, -72.1195, 18.6611)
        params["CROP"] = True
        AutoRoute(params).run()

        if not os.path.exists(output):
            output = output.replace(".vrt", ".tif")

        with gdal.Open(output) as out_ds, gdal.Open(validation) as val_ds:
            assert out_ds is not None
            assert (out_ds.ReadAsArray() - val_ds.ReadAsArray()).max() < 9.2 # Because cells are shifted, values shift as well oh so slightly, which a vrt does not pick up
            assert np.allclose(out_ds.GetGeoTransform(), val_ds.GetGeoTransform(), atol=1e-6)
            assert out_ds.GetProjection() == val_ds.GetProjection()
        tearDown(output)

    def test_flowfile():
        params = {"OVERWRITE": True,
               "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
               "DATA_DIR": "test_ar_data",
               "DEM_NAME": "test_dem", 
               "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"),
               "STREAM_NAME": "test_strm", 
               "STREAM_ID": "LINKNO",
               "FLOOD_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flowfile.csv",),
        }
        output = os.path.join("test_ar_data","flow_files","N18W073_FABDEM_V1-2__flow.txt")
        validation = os.path.join("tests","test_data","validation","flowfile","N18W073_FABDEM_V1-2__strm__flow.txt")

        AutoRoute(params).run()

        assert os.path.exists(output)
        out_df = pd.read_csv(output)
        val_df = pd.read_csv(validation)
        pd.testing.assert_frame_equal(out_df, val_df, check_exact=False)
        tearDown(output)

    @pytest.mark.skipif(sys.platform == "darwin", reason="Not supported on macOS")
    @pytest.mark.skipif(not os.path.exists(AUTOROUTE_EXE), reason="AutoRoute exe not found")
    def test_autoroute():
        params = {"OVERWRITE": True,
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
                "MANNINGS_TABLE": os.path.join("tests","test_data","mannings_table","mannings.txt"),
                 "DISABLE_PBAR": True}
        output = os.path.join("test_ar_data","vdts","N18W073_FABDEM_V1-2__vdt.txt")
        validation = os.path.join("tests","test_data","validation","vdts","simple_inputs.txt")
        
        run_autoroute(params, output)
        with open(output) as f:
            out = f.read()
        with open(validation) as f:
            val = f.read()
        assert out == val
        tearDown(output)

#     def test_AutoRoute_lots_of_params(self):
#         params["RAPID_Subtract_BaseFlow"] = True
#         params["VDT"] = "test_ar_data"
#         params["num_iterations"] = 5
#         params["convert_cfs_to_cms"] = True
#         params["x_distance"] = 100
#         params["q_limit"] = 1.01
#         params["direction_distance"] = 10
#         params["slope_distance"] = 10
#         params["weight_angles"] = 1
#         params["use_prev_d_4_xs"] = 0
#         params["adjust_flow"] = 0.9
#         params["row_start"] = 0
#         params["row_end"] = 1000
#         params["degree_manip"] = 6.1
#         params["degree_interval"] = 1.5
#         params["man_n"] = 0.01
#         params["low_spot_distance"] = 30
#         params["low_spot_is_meters"] = True
#         params["low_spot_use_box"] = True
#         params["box_size"] = 5
#         params["find_flat"] = True
#         params["ar_bathy_file"] = 'test_ar_data'
#         params["bathy_alpha"] = 0.001
#         params["bathy_method"] = 'Trapezoidal'
#         params["bathy_x_max_depth"] = 0.3
#         params["bathy_y_shallow"] =  0.3
#         output = os.path.join("test_ar_data","N18W073_FABDEM_V1-2__ar_bathy.tif")
#         validation = os.path.join("tests","test_data","validation","bathy","ar_bathy_all_params.tif")

#         global run_floodspreader
#         run_floodspreader = False
#         AutoRouteHandler(params).run()

#         assertTrue(os.path.exists(output), f"File does not exist: {output}")
#         out_ds = gdal.Open(output)
#         assertIsNotNone(out_ds, "Problem opening file")
#         val_ds = gdal.Open(validation)
#         assertTrue(np.array_equal(out_ds.ReadAsArray(), val_ds.ReadAsArray()), "Arrays are not equal")
#         assertTrue(np.isclose((np.array(out_ds.GetGeoTransform()) - np.array(val_ds.GetGeoTransform())).max(), 0), "GeoTransform is not equal")
#         assertEqual(out_ds.GetProjection(), val_ds.GetProjection(), "Projection is not equal")

#         out_ds = None
#         os.remove(output)

#         output = os.path.join("test_ar_data","N18W073_FABDEM_V1-2__vdt.txt")
#         validation = os.path.join("tests","test_data","validation","vdts","all_params.txt")
        
#         assertTrue(os.path.exists(output), f"File does not exist: {output}")
#         with open(output) as f:
#             out = f.read()
#         with open(validation) as f:
#             val = f.read()
#         assertEqual(out, val, "VDT files are not equal")
#         run_floodspreader = True

# @unittest.skip
# @unittest.skipIf(sys.platform != "win32" and sys.platform != "linux", "Runs only on windows or linux")
# @unittest.skipIf(not os.path.exists(os.path.join(AUTOROUTE_EXE)) or not os.path.exists(FLOODSPREADER_EXE), "AutoRoute and/or FloodSpreader not found")
# @unittest.skipIf(not run_floodspreader, "Autoroute failed, not running FloodSpreader")
# class TestFloodSpreader(unittest.TestCase):
#     def setUp(self) -> None:
#         params = {"OVERWRITE": True,
#               "DATA_DIR": "test_ar_data",
#               "DEM_FOLDER": os.path.join("tests","test_data","DEMs","single_4326"),
#               "BUFFER_FILES": False, 
#               "DEM_NAME": "test_dem", 
#               "STREAM_NETWORK_FOLDER": os.path.join("tests","test_data","streamlines","single_parquet_4326"), 
#               "STREAM_NAME": "test_strm", 
#               "STREAM_ID": "LINKNO",
#               "LAND_USE_FOLDER": os.path.join("tests","test_data","LUs","single_4326"),
#               "LAND_USE_NAME": "test_land",
#               "SIMULATION_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flows.csv",),
#               "SIMULATION_ID_COLUMN": "LINKNO",
#               "SIMULATION_FLOW_COLUMN": "max",
#               "BASE_FLOW_COLUMN": "flow",
#               "FLOOD_FLOWFILE":  os.path.join("tests","test_data","flow_files","v2_flowfile.csv",),
#               "AUTOROUTE": AUTOROUTE_EXE,
#               "FLOODSPREADER": FLOODSPREADER_EXE,
#               "AUTOROUTE_CONDA_ENV": "autoroute",
#               "MANNINGS_TABLE": os.path.join("tests","test_data","mannings_table","mannings.txt"),
#                "FLOOD_MAP":os.path.join("test_ar_data","simple_map"),
#                 "DISABLE_PBAR": True}
#         output = os.path.join("test_ar_data","simple_map","N18W073_FABDEM_V1-2__flood.tif")
#         validation = os.path.join("tests","test_data","validation","maps","simple_flood.tif")
#         tearDown()
        
#     def tearDown(self) -> None:
#         try:
#             {os.remove(m) for m in glob.glob(os.path.join("test_ar_data","*bathy.tif"))}
#             if output and os.path.exists(output): os.remove(output) 
#         except PermissionError:
#             print(f"\nCould not remove {output}\n")

#     def test_FloodSpreader(self):
#         AutoRouteHandler(params).run()
        
#         out_ds = gdal.Open(output)
#         assertIsNotNone(out_ds, "Problem opening file")
#         val_ds = gdal.Open(validation)
#         out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
#         out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
#         out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
#         out_ds = None
#         assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
#         assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
#         assertEqual(out_proj, val_proj, "Projection is not equal")
        

#     def test_FloodSpreader_lots_of_params(self):
#         params["ar_bathy_file"] = 'test_ar_data'
#         params["bathy_alpha"] = 0.001
#         params["bathy_method"] = 'Trapezoidal'

#         params["omit_outliers"] = 'Smooth Water Surface Elevation'
#         params["wse_search_dist"] = 15
#         params["wse_threshold"] = 0.2
#         params["wse_remove_three"] = True
#         params["specify_depth"] = 0
#         params["twd_factor"] = 2
#         params["only_streams"] = False
#         params["use_ar_top_widths"] = True
#         params["flood_local"] = True
#         params["fs_bathy_file"] = os.path.join("test_ar_data","many_bathy")
#         params["fs_bathy_smooth_method"] = ''
#         params["bathy_twd_factor"] = 1

#         output = os.path.join("test_ar_data","many_bathy","N18W073_FABDEM_V1-2__fs_bathy.tif")
#         validation = os.path.join("tests","test_data","validation","bathy","fs_bathy_all_params.tif")

#         AutoRouteHandler(params).run()

#         assertTrue(os.path.exists(output), f"File does not exist: {output}")
#         out_ds = gdal.Open(output)
#         assertIsNotNone(out_ds, "Problem opening file")
#         val_ds = gdal.Open(validation)
#         out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
#         out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
#         out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
#         out_ds = None
#         val_ds = None
#         assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
#         assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
#         assertEqual(out_proj, val_proj, "Projection is not equal")

#         os.remove(output)

#         output = os.path.join("test_ar_data","simple_map","N18W073_FABDEM_V1-2__flood.tif")
#         validation = os.path.join("tests","test_data","validation","maps","all_params_flood.tif")

#         out_ds = gdal.Open(output)
#         assertIsNotNone(out_ds, "Problem opening file")
#         val_ds = gdal.Open(validation)
#         out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
#         out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
#         out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
#         out_ds = None
#         val_ds = None
#         assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
#         assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
#         assertEqual(out_proj, val_proj, "Projection is not equal")

#     def test_buffer(self):
#         params["DEM_FOLDER"] = os.path.join("tests","test_data","DEMs","tiles"),
#         params["BUFFER_FILES"] = True

#         AutoRouteHandler(params).run()
        
#         out_ds = gdal.Open(output)
#         assertIsNotNone(out_ds, "Problem opening file")
#         val_ds = gdal.Open(validation)
#         out_arr, val_arr = out_ds.ReadAsArray(), val_ds.ReadAsArray()
#         out_geo, val_geo = out_ds.GetGeoTransform(), val_ds.GetGeoTransform()
#         out_proj, val_proj = out_ds.GetProjection(), val_ds.GetProjection()
#         out_ds = None
#         assertTrue(np.array_equal(out_arr, val_arr), "Arrays are not equal")
#         assertTrue(np.isclose((np.array(out_geo) - np.array(val_geo)).max(), 0), "GeoTransform is not equal")
#         assertEqual(out_proj, val_proj, "Projection is not equal")

if __name__ == '__main__':
    test_crop_dem()

