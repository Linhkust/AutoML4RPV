from shiny import App, ui, reactive, Inputs, Outputs, Session, render
import pandas as pd
from ipyleaflet import Map, GeoData, basemaps, FullScreenControl, GeoJSON
from shinywidgets import output_widget, render_widget
import faicons as fa
import json
import geopandas as gpd
from data_preparation import DP, gpt_transformation, address_transformation
import webbrowser
from pandas import json_normalize
from feature_engineering import feature_type, stream_feature_generation, unzip, boundary_import, statistics
import plotly.express as px
import numpy as np
from model_training import fit, benchmark_tpot, benchmark_h2o, benchmark_flaml, benchmark_pycaret, benchmark_mjar, benchmark
from model_evaluation import best_pipelines, pipelines
from feature_importance import feature_importance_
import joblib
import os
from sklearn.model_selection import train_test_split

ICONS = {
    "pipeline": fa.icon_svg("timeline"),
    "time": fa.icon_svg("clock"),
    'target': fa.icon_svg('ranking-star'),
    'play': fa.icon_svg('play'),
    'bots': fa.icon_svg('bots')
}

app_ui = ui.page_navbar(

    # Data collection
    ui.nav_panel("Data Collection",
                 ui.page_sidebar(
                     ui.sidebar(
                         ui.input_file('boundary', 'Upload Cartographic Data', accept=['.geojson'],
                                       placeholder='No file selected'),

                         # osm data link
                         ui.output_ui("osm_link"),
                         ui.input_action_button('osm_download', 'Download OSM Feature Data', class_="btn-success"),
                         width=400, open='always',
                     ),

                     ui.layout_columns(

                         # map
                         ui.card(
                             ui.input_switch("type_switch", "OSM Feature Layer", False),
                             output_widget('osm_display'),
                             ui.card_footer(ui.input_action_link('boundary_link', 'OSM Boundary Website')),
                             height='1000px',
                         ), ),

                     ui.layout_columns(
                         # data preview
                         ui.card(
                             ui.output_data_frame("spatial_preview")
                         ),
                     ),
                 )
                 ),

    # Data Preparation
    ui.nav_panel("Data Preparation",
                 ui.page_sidebar(
                     ui.sidebar(
                         ui.accordion(
                             ui.accordion_panel("Step 1: Housing Data Upload",
                                                ui.input_file("raw",
                                                              "Upload your data",
                                                              accept=['.csv'],
                                                              placeholder='No file selected'),
                                                ),

                             ui.accordion_panel("Step 2: Column Selection",
                                                ui.input_selectize('housing', 'Physical & Transaction Attributes',
                                                                   choices=[], multiple=True,
                                                                   options={"plugins": ["clear_button"]}),
                                                ui.input_selectize('address', 'Property Address', choices=[],
                                                                   multiple=True,
                                                                   options={"plugins": ["clear_button"]}),
                                                ui.input_selectize('price', 'Price', choices=[], multiple=True,
                                                                   options={"plugins": ["clear_button"]}),
                                                ),

                             ui.accordion_panel("Step 3: Ordinal Columns Transformation",
                                                ui.input_selectize('ordinal', 'Ordinal Columns (if any)',
                                                                   choices=[],
                                                                   multiple=True),
                                                ui.input_action_button('add_ranking', 'Add ranking', class_="btn-success", disabled=True),
                                                ui.input_action_button('remove_ranking', 'Remove ranking', class_="btn btn-warning", disabled=True),
                                                ),

                             ui.accordion_panel("Step 4: Column Reorganization",
                                                ui.input_selectize('reorganize', 'Column to be transformed (if any)',
                                                                   choices=[],
                                                                   multiple=True),

                                                ui.input_action_button('add_codes', 'Add Codes',
                                                                       class_="btn-success", disabled=True),
                                                ui.input_action_button('remove_codes', 'Remove Codes',
                                                                       class_="btn btn-warning", disabled=True),
                                                ui.input_action_button("login", "Easycoding Chatbot", class_="btn btn-info", icon='🤖', disabled=True)
                                                ),

                             ui.accordion_panel("Step 5: Geocoding Configuration",
                                                ui.input_switch("api", "Use Google Map for Geocoding", False),
                                                ui.output_ui("api_key"),
                                                ui.input_selectize('estate', 'Estate Name', choices=[], multiple=True,
                                                                   options={"plugins": ["clear_button"]}),
                                                ui.input_selectize('street', 'Estate Address', choices=[],
                                                                   multiple=True,
                                                                   options={"plugins": ["clear_button"]}),
                                                ui.input_text("city", "City/Country Name")
                                                ),

                             id="dp",
                             open="Housing Data Upload",
                         ),

                         ui.input_task_button('preparation', 'Processing WITHOUT geocoding'),
                         ui.output_ui('preparation_geocoding'),
                         ui.download_button('preparation_download', 'Download', class_="btn-success"),
                         width=500, open='always',
                     ),

                     # Data statistics
                     ui.layout_columns(
                         ui.value_box("Number of Total Records", ui.output_ui("total_records"), theme="bg-blue", ),
                         ui.value_box("Number of Total Columns", ui.output_ui("total_columns"), theme="bg-blue", ),
                         fill=False,
                     ),

                     # Data Preview
                     ui.layout_columns(
                         ui.card(
                             ui.card_header('Data Preview'),
                             ui.output_data_frame("preparation_table"),
                             full_screen=True,
                             height='750px',
                         ),
                     ),

                    ui.layout_columns(
                        ui.card(
                            ui.card_header('Data Preview After Geocoding'),
                            ui.output_data_frame('compute'),
                            full_screen=True,
                            height='750px',
                        ),
                     ),

                 )
                 ),

    # Feature Engineering
    ui.nav_panel("Feature Engineering",
                 ui.page_sidebar(
                     ui.sidebar(
                         ui.accordion(
                             ui.accordion_panel("Step 1: Data Upload",
                                                ui.input_file("osm_feature_zip",
                                                              "OSM Feature Folder Path",
                                                              accept=['.zip'],
                                                              placeholder='No file selected'),

                                                ui.input_file("feature_boundary", "Boundary Data Upload",
                                                              accept=['.geojson'],
                                                              placeholder='No file selected'),

                                                ui.input_file("clean_data",
                                                              "Processed and Geocoded Housing Data Upload",
                                                              accept=['.csv'],
                                                              placeholder='No file selected'),
                                                ),

                             ui.accordion_panel("Step 2: OSM Feature Generation Configuration",

                                                # Point attributes
                                                ui.card(
                                                    ui.card_header('Point Attributes'),
                                                    ui.input_selectize('point_features', 'Select Point Types',
                                                                       choices=[],
                                                                       multiple=True),
                                                    ui.input_numeric("point_distance", "Select Distance Range", 1200, min=400,
                                                                     max=1600, step=80),
                                                    ui.input_radio_buttons('point_configuration',
                                                                           'Area/Distance Type for Density/Diversity/Accessibility Calculation',
                                                                           {
                                                                               'network': 'Isodistance/Network Distance',
                                                                               'straight': 'Circular Area/Euclidean Distance'}),
                                                ),

                                                # Line attributes
                                                ui.card(
                                                    ui.card_header('Line Attributes'),
                                                    ui.input_selectize('line_features', 'Select Line Types', choices=[],
                                                                       multiple=True),
                                                ),

                                                # Polygon attributes
                                                ui.card(
                                                    ui.card_header('Polygon Attributes'),
                                                    ui.input_selectize('polygon_features', 'Select Polygon Types',
                                                                       choices=[],
                                                                       multiple=True),
                                                    ui.input_radio_buttons('polygon_distance',
                                                                           'Area/Distance Type for Accessibility and Building Density',
                                                                           {'network': 'Isodistance/Network Distance',
                                                                            'straight': 'Circular Area/Euclidean Distance'}),
                                                    ui.input_numeric("polygon_distance", "Select Distance Range for Building Density Calculation", 400,
                                                                     min=0,
                                                                     max=1200, step=80),
                                                ),
                                                ),

                             id="fe",
                             open="Spatial Data Upload",
                         ),

                         ui.input_task_button('feature_processing', 'Generate Features'),
                         ui.input_task_button('feature_preview', 'Preview'),
                         ui.download_button('feature_download', 'Download', class_="btn-success"),
                         width=500, open='always',
                     ),

                     # Feature data preview
                     ui.layout_columns(
                         ui.value_box("Number of Total Records", ui.output_ui("feature_records"), theme="bg-blue"),
                         ui.value_box("Number of Total Features", ui.output_ui("feature_columns"), theme="bg-blue"),
                         fill=False,
                     ),

                     # interactive map
                     ui.layout_columns(
                         ui.card(
                             ui.card_header('Map Display'),
                             ui.input_select('map_features', 'Selected Features', choices=[], multiple=False),
                             ui.input_task_button('map_show', 'Show', width='300px'),
                             output_widget("feature_map"),
                             height='800px',
                         ),
                     ),

                     # data preview
                     ui.layout_columns(
                         ui.card(
                             ui.card_header('Data Preview'),
                             ui.output_data_frame("feature_table"),
                             full_screen=True,
                             height='600px',
                         )),

                    # data statistics
                    # ui.layout_columns(
                    #      ui.card(
                    #          ui.card_header('Data Statistics'),
                    #          ui.output_data_frame("statistics"),
                    #          full_screen=True,
                    #          height='600px',
                    #      ),
                    #  ),

                     # Data statistics and correlation matrix
                     # ui.layout_columns(
                     #     ui.card(
                     #         ui.card_header('Correlation Matrix'),
                     #         output_widget("correlation"),
                     #         full_screen=True,
                     #         height='800px',
                     #     ),
                     # ),
                 ),
                 ),

    # Model Generation
    ui.nav_panel("Model Pipeline Generation & Evaluation",
                 ui.page_sidebar(
                     ui.sidebar(
                         # data Upload
                         ui.card(
                             ui.card_header('Step 1: Data Upload'),
                             ui.input_file("final", "Upload your data", accept='.csv', placeholder='No file selected'),
                             ui.input_select('target_col', 'Target column', choices=[]),
                         ),

                         # train test split
                         ui.card(
                             ui.card_header('Step 2: Model Training Configuration'),
                             ui.input_slider("train_test", "Training data size", 0, 1, 0.7, step=0.05),
                             ui.input_select("mrmr", "MRMR feature selection configuration",
                                                {'best_quality':'best_quality',
                                                 'high_quality':'high_quality',
                                                 'good_quality':'good_quality'},
                                                selected='good_quality'),
                             ui.input_switch("automl", "Use Other AutoML Frameworks", False),
                             ui.output_ui("automl_choice"),
                         ),

                         # configuration of AutoML frameworks
                         ui.output_ui('automl_configuration'),
                         ui.input_task_button('model_training', 'Model Train'),
                         ui.download_button('training_download', 'Download', class_="btn-success"),
                         width=500, open='always'
                     ),

                     # Training results
                     ui.layout_columns(
                         ui.value_box("Number of Total Pipelines", ui.output_ui("pipelines_num"),
                                      showcase=ICONS['pipeline']),
                         ui.value_box("Total Training Time", ui.output_ui("training_time"),
                                      showcase=ICONS['time']),
                         ui.value_box("Best Pipeline", ui.output_ui("best_pipeline"),
                                      showcase=ICONS['target']),
                         fill=False,
                     ),

                     # model performance leaderboard
                     ui.layout_columns(
                         ui.card(
                             ui.card_header('Model Performance Leaderboard'),
                             ui.output_data_frame('leaderboard'),
                             height='600px'
                         )
                     ),

                     # taylor diagram
                     # ui.layout_columns(
                     #     # Taylor Diagram
                     #     ui.card(
                     #         ui.card_header(ui.input_selectize(
                     #             "pipeline_selection",
                     #             "Selected pipelines for Taylor diagram:",
                     #             choices=[],
                     #             multiple=True, width='700px'
                     #         ),
                     #         ),
                     #         ui.output_plot('taylor_plot'), height='900px'
                     #     ),
                     # ),

                     # model performance plot
                     # ui.layout_columns(
                     #
                     #     # train set regression plot
                     #     ui.card(
                     #         ui.card_header('Regression plot of the best pipeline (train set)'),
                     #         ui.output_plot('regression_plot_train'), height='600px'
                     #     ),
                     #
                     #     # best model performance
                     #     ui.card(
                     #         ui.card_header('Regression plot of the best pipeline (test set)'),
                     #         ui.output_plot('regression_plot_test'), height='600px'
                     #     ),
                     # )
                 )
                 ),

    # XAI
    ui.nav_panel("Explainable Artificial Intelligence",
                 ui.page_sidebar(
                     ui.sidebar(

                         # Data upload
                         ui.card(
                             ui.card_header('Step 1: Data Upload'),
                             ui.input_file("xai_data", "Upload the data", accept=['.csv'],
                                           placeholder='No file selected'),
                             ui.input_select('xai_target_col', 'Target column', choices=[]),
                         ),
                            ui.card(
                                            ui.card_header('Step 2: Model Upload'),
                                            ui.input_file("pkl_upload", "Upload ML model", accept=['.pkl'],
                                                          placeholder='No file selected'),
                                        ),
                         # # Model upload
                         # ui.input_switch("pkl", "Use your own model", False),
                         # ui.output_ui("pkl_upload"),
                         # Model explain
                         ui.input_task_button('model_explain', 'Model Explain'),
                         ui.download_button('xai_download', 'Download', class_="btn-success"),
                         ui.HTML("""
                                         <p>
                                         <b>Interpretation of XAI plots</b> <br>
                                          <b>[1] Permutation Feature Importance:</b> PFI plot ranks features by their impact on model performance (higher bars = more important features), 
                                          showing which features most degrade predictions when shuffled. <br>
                                          
                                         <b>[2] SHAP summary plot:</b> The summary plot combines feature importance with feature effects. 
                                         Each point on the summary plot is a Shapley value for a feature and an instance. 
                                         The position on the y-axis is determined by the feature and on the x-axis by the Shapley value. <br>
                                         
                                         <b>[3] Partial Dependence Plot:</b> The PDP shows the marginal effect one or two features have on the predicted outcome of a machine learning model <br>
                                         
                                         <b>[4] SHAP dependence plot: </b> The SHAP dependence plot reveals how a single feature's values impact model predictions (vertical SHAP values) 
                                         while uncovering potential interactions with another feature (color-coded). <br>
                                         
                                         <b>[5] SHAP values: </b> SHAP values quantify the contribution of each feature to a model's prediction for a specific instance, 
                                         showing both magnitude (importance) and direction (positive/negative impact).<br>
                                         
                                         </p>
                                         """
                                 ),

                         width=500, open='always'
                     ),

                     ui.navset_card_tab(
                         ui.nav_panel("Permutation Feature Importance",
                                       ui.output_plot('pfi')),

                         ui.nav_panel("SHAP Summary Plot",
                                      ui.output_plot('shap_global')),

                         ui.nav_panel("Dependence Plot & SHAP Values",
                                      ui.input_select('xai_feature', 'Selected Features',
                                     choices=[],
                                     multiple=False,
                                     ),
                                      ui.input_task_button("xai_plot", "Plot", width='300px'),
                                      ui.layout_columns(
                                          # PDP plot
                                          ui.card(
                                              ui.card_header('Partial Dependence Plot'),
                                              ui.output_plot('pdp'),
                                              height='500px'),

                                          # SHAP depedence plot
                                          ui.card(
                                              ui.card_header('SHAP Dependence Plot'),
                                              ui.output_plot('shap_dp'),
                                              height='500px'),
                                      ),
                                    ui.card(
                                 ui.card_header('Spatial distribution of SHAP values'),
                                 output_widget("shap_map"),
                                 height='700px',
                                             )
                                      ),
                         id="tab",
                     ),

# Global feature importance analysis results
#                      ui.layout_columns(
#                          # Permutation feature importance
#                          ui.card(
#                              ui.card_header('Permutation Feature Importance'),
#                              ui.output_plot('pfi'),
#                              height='700px',
#                          ),
#                      ),
#
#                      ui.layout_columns(
#                          # SHAP analysis
#                          ui.card(
#                              ui.card_header('SHAP Summary Plot'),
#                              ui.output_plot('shap_global'),
#                              height='700px'
#                          ),
#                      ),
#
#                      # Local feature importance analysis results
#                      ui.input_select('xai_feature', 'Selected Features',
#                                      choices=[],
#                                      multiple=False,
#                                      ),
#                      ui.input_task_button("xai_plot", "Plot"),
#
#                      ui.layout_columns(
#                          # PDP plot
#                          ui.card(
#                              ui.card_header('Partial Dependence Plot'),
#                              ui.output_plot('pdp'),
#                              height='500px'),
#
#                          # SHAP depedence plot
#                          ui.card(
#                              ui.card_header('SHAP Dependence Plot'),
#                              ui.output_plot('shap_dp'),
#                              height='500px'),
#                      ),
#
#                      ui.card(
#                          ui.card_header('Spatial distribution of SHAP values'),
#                          output_widget("shap_map"),
#                          height='800px',
#                      ),

                 ),
                 ),

    title="[AutoML4RPV]",
    id="page",
)


def server(input, output, session):

    '''
    Data Collection
    '''
    @render_widget
    def osm_display():
        file = input.boundary()
        value = input.type_switch()
        m = Map(center=(22.33601623133733, 114.26386127778946),
                basemap=basemaps.CartoDB.Positron,
                zoom=11,
                scroll_wheel_zoom=True)

        # boundary data
        if value is False:
            if file is None:
                return m
            else:
                print(file[0]["datapath"])
                data = gpd.read_file(file[0]["datapath"])
                if len(data) > 0:
                    m = Map(center=(data.centroid[0].y, data.centroid[0].x), basemap=basemaps.CartoDB.Positron, zoom=11,
                            scroll_wheel_zoom=True)
                else:
                    m = Map(center=(data.centroid.y, data.centroid.x), basemap=basemaps.CartoDB.Positron, zoom=11,
                            scroll_wheel_zoom=True)

                geo_data = GeoData(geo_dataframe=data,
                                   style={'color': 'black', 'fillColor': '#3366cc', 'opacity': 0.05, 'weight': 1.9,
                                          'dashArray': '2', 'fillOpacity': 0.6},
                                   hover_style={'fillColor': 'red', 'fillOpacity': 0.2}, )
                m.add(geo_data)
                m.add(FullScreenControl())
                return m

        # OSM feature switch
        else:
            m = Map(center=(35.13751256247316, -40.64777487898175),
                    basemap=basemaps.CartoDB.Positron,
                    zoom=3,
                    scroll_wheel_zoom=True)

            rows = spatial_preview.input_cell_selection()["rows"]

            if rows:
                with open('index-v1.json', 'r') as f:
                    data = json.load(f)

                selected_data = data['features'][rows[0]]
                geo_json = GeoJSON(
                    data=selected_data,
                    style={
                        'opacity': 5, 'dashArray': '9', 'fillOpacity': 0.5, 'weight': 1
                    },
                    hover_style={
                        'color': 'blue', 'dashArray': '0', 'fillOpacity': 0.6
                    },
                )
                m.add(geo_json)
                m.add(FullScreenControl())
                return m
            else:
                m.add(FullScreenControl())
                return m

    @render.data_frame
    def spatial_preview():
        file = input.boundary()
        value = input.type_switch()
        if value is False:
            if file is None:
                return render.DataGrid(pd.DataFrame())
            else:
                with open(file[0]["datapath"], 'r', encoding='utf_8_sig') as f:
                    data = json.load(f)
                    df = json_normalize(data['features'])
                    return render.DataGrid(df, filters=True, selection_mode='row')
        else:
            with open('index-v1.json', 'r', encoding='utf_8_sig') as f:
                osm_data = json.load(f)
            osm_df = json_normalize(osm_data['features'])
            return render.DataGrid(osm_df, filters=True, selection_mode='row')

    @render.ui
    def osm_link():
        value = input.type_switch()
        if value is True:
            with open('index-v1.json', 'r') as f:
                osm_data = json.load(f)
            osm_df = json_normalize(osm_data['features'])
            rows = spatial_preview.input_cell_selection()["rows"]
            if rows:
                return osm_df.loc[rows, 'properties.urls.shp']
            else:
                return "OSM data not selected!"

    @reactive.effect
    @reactive.event(input.osm_download)
    def _():
        value = input.type_switch()
        if value is True:
            with open('index-v1.json', 'r') as f:
                osm_data = json.load(f)
            osm_df = json_normalize(osm_data['features'])
            rows = spatial_preview.input_cell_selection()["rows"]
            if rows:
                webbrowser.open(osm_df.loc[rows, 'properties.urls.shp'], new=2)

    @reactive.effect
    @reactive.event(input.boundary_link)
    def _():
        webbrowser.open('https://osm-boundaries.com/map', new=2)


    ''' Data Preparation'''
    @reactive.calc
    def raw_csv():
        file = input.raw()
        empty = []
        if file is None:
            return pd.DataFrame(), empty
        else:
            df = pd.read_csv(file[0]["datapath"])
            choice_list = {}
            for column in df.columns:
                choice_list[column] = column
            return df, choice_list

    def clean_data():
        file = input.raw()

        if file is None:
            return pd.DataFrame()
        else:

            # Step 2 has no information, then preview raw data
            if len(input.housing()) * len(input.address()) * len(input.price())  == 0:
                return raw_csv()[0]

            # start data preparation process
            else:
                # data clean and coding
                data = raw_csv()[0]

                # data clean and coding
                if len(input.ordinal()) != 0:
                    ranking_list = []
                    for id, column in enumerate(input.ordinal()):
                        ranking_list.append(eval('input.ranking_results{}()'.format(str(id))))
                else:
                    ranking_list = None

                # Reorganization
                if len(input.reorganize()) != 0:
                    codes_list = []
                    for id, column in enumerate(input.reorganize()):
                        codes_list.append([eval('input.codes{}()'.format(str(id)))])
                else:
                    codes_list = None

                data_processor = DP(housing_columns=list(input.housing()),
                                    address_columns=list(input.address()),
                                    price_column=list(input.price()),
                                    property_data=data,
                                    ordinal_columns=input.ordinal(),
                                    ordinal_rankings=ranking_list,
                                    reorganize_columns=input.reorganize(),
                                    reorganize_codes=codes_list
                                    )

                data = data_processor.clean_data()
                return data


    # update ordinal columns
    @reactive.effect
    def update_ordinal():
        if input.raw() is not None:
            ui.update_selectize(
                'ordinal',
                choices=raw_csv()[1])

    # add ranking results of selected columns
    @reactive.effect
    @reactive.event(input.add_ranking)
    def add_ranking():
        if input.raw() is not None:
            if input.ordinal():
                for id, column in enumerate(input.ordinal()):
                    ui.insert_ui(ui.input_selectize('ranking_results'+str(id),
                                                    'Ranking results of ' + column,
                                                    multiple=True,
                                                    choices=raw_csv()[0][column].unique().tolist(),
                                                    options={"plugins": ["clear_button"]}),
                                 selector='#add_ranking',
                                 where='afterEnd')

    # add codes
    @reactive.effect
    @reactive.event(input.add_codes)
    def add_codes():
        if input.raw() is not None:
            if input.reorganize():
                for id, column in enumerate(input.reorganize()):
                    ui.insert_ui(ui.input_text_area('codes' + str(id),
                                                    'Python codes of ' + column),
                                 selector='#login', where='afterEnd')


    # remove ranking results of selected columns
    @reactive.effect
    @reactive.event(input.remove_ranking)
    def remove_ranking():
        if input.raw() is not None:
            if input.ordinal():
                for id, column in enumerate(input.ordinal()):
                    ui.remove_ui(selector="div:has(> #ranking_results{}-label)".format(str(id)))
                    ui.remove_ui(selector="div:has(> #ranking_results{})".format(str(id)))


    @reactive.effect
    @reactive.event(input.remove_codes)
    def remove_codes():
        if input.raw() is not None:
            if input.reorganize():
                for id, column in enumerate(input.reorganize()):
                    ui.remove_ui(selector="div:has(> #codes{}-label)".format(str(id)))
                    ui.remove_ui(selector="div:has(> #codes{})".format(str(id)))

    @reactive.effect
    @reactive.event(input.ordinal)
    def update_add_ranking_button():
        if len(input.ordinal()) != 0:
            ui.update_action_button("add_ranking", disabled=False)
            ui.update_action_button("remove_ranking", disabled=False)
        else:
            ui.update_action_button("add_ranking", disabled=True)
            ui.update_action_button("remove_ranking", disabled=True)

    @reactive.effect
    @reactive.event(input.reorganize)
    def update_add_codes_button():
        if len(input.reorganize()) != 0:
            ui.update_action_button("add_codes", disabled=False)
            ui.update_action_button("remove_codes", disabled=False)
            ui.update_action_button("login", disabled=False)
        else:
            ui.update_action_button("add_codes", disabled=True)
            ui.update_action_button("remove_codes", disabled=True)
            ui.update_action_button("login", disabled=True)


    # EasyCoding ChatBOT
    @reactive.effect
    @reactive.event(input.login)
    def _():
        # Chat Bot
        message = {
            "content": "**Hello! Can you describe the function used for column reorganization?**",
            "role": "assistant"
        }

        chat = ui.Chat(id="chat", messages=[message])

        @chat.on_user_submit
        async def _():
            await chat.append_message(gpt_transformation(chat.user_input()))

        m = ui.modal(
            ui.chat_ui('chat'),
            title="EasyCoding Chatbot",
            easy_close=True,
            footer=None,
            size='xl',
        )
        ui.modal_show(m)

    # processing button with geocoding disable
    @render.ui
    @reactive.event(input.city)
    def preparation_geocoding():
        if input.city():
            return ui.input_task_button("preparation_geocoding_btn", 'Processing WITH geocoding')

    @render.ui
    @reactive.event(input.preparation, ignore_none=False)
    def total_records():
        return len(clean_data())

    @render.ui
    @reactive.event(input.preparation, ignore_none=False)
    def total_columns():
        return len(clean_data().columns)

    @reactive.effect
    def _():
        file = input.raw()
        if file is None:
            for id in ['housing',
                       'address',
                       'reorganize',
                       'ordinal',
                       'price',
                       'estate',
                       'street'
                       ]:
                ui.update_selectize(
                    id,
                    choices=[], )
        else:
            for id in ['housing',
                       'address',
                       'price']:
                ui.update_selectize(
                    id,
                    choices=raw_csv()[1], )

    @reactive.effect
    def _():
        for id in [
            'reorganize',
            'ordinal']:
            ui.update_selectize(
                id,
                choices=list(input.housing())+list(input.address()))

        for id in ['estate',
                   'street']:
            ui.update_selectize(
                id,
                choices=list(input.address()))

    @render.ui
    @reactive.event(input.api)
    def api_key():
        if input.api() is True:
            return ui.input_text("api_key", 'API Key')

    @render.data_frame
    @reactive.event(input.preparation, ignore_none=False)
    def preparation_table():
        return render.DataGrid(clean_data())

    @render.download()
    def preparation_download():
        file = input.raw()
        path = file[0]["datapath"]
        download_path = os.path.dirname(path)+'_preparation.csv'
        clean_data().to_csv(download_path, index=False)
        return download_path


    @render.data_frame
    @reactive.event(input.preparation_geocoding_btn)
    def compute():
        data = clean_data()
        if input.api() is True:
            geocoding_results = address_transformation(data=data,
                                                       property=input.estate()[0],
                                                       street=input.street()[0],
                                                       city=input.city(),
                                                       google_api_key=input.api_key()
                                                       )
        else:
            geocoding_results = address_transformation(data=data,
                                                       property=input.estate()[0],
                                                       street=input.street()[0],
                                                       city=input.city()
                                                      )

        return render.DataGrid(geocoding_results)


    '''Feature Engineering'''

    @reactive.effect
    def _():
        if input.osm_feature_zip() is not None:
            unzip(input.osm_feature_zip()[0]["datapath"]) # unzip the file

    @reactive.effect
    def _():
        if input.feature_boundary() is not None:
            # update feature selection options
            zip_path = input.osm_feature_zip()[0]["datapath"]
            zip_name = input.osm_feature_zip()[0]["name"]
            osm_path = os.path.join(os.path.dirname(zip_path), zip_name.split('.')[0])

            if input.feature_boundary() is not None:
                (point_features,
                 line_features,
                 polygon_features) = feature_type(osm_path=osm_path,
                                                     boundary=boundary_import(
                                                        boundary_file=input.feature_boundary()[0]["datapath"]))

                ui.update_selectize(
                    'point_features',
                    choices=point_features['fclass'].unique().tolist())

                ui.update_selectize(
                    'line_features',
                    choices=line_features['fclass'].unique().tolist())

                ui.update_selectize(
                    'polygon_features',
                    choices=polygon_features[1]['fclass'].unique().tolist())


    @render.data_frame
    @reactive.event(input.feature_processing, ignore_none=False)
    def feature_table():
        if input.clean_data() is not None:
            zip_path = input.osm_feature_zip()[0]["datapath"]
            zip_name = input.osm_feature_zip()[0]["name"]
            osm_path = os.path.join(os.path.dirname(zip_path), zip_name.split('.')[0])

            file = input.clean_data()
            path = file[0]["datapath"]
            download_path = os.path.dirname(path) + '_features.csv'

            # run the main func
            stream_feature_generation(data=pd.read_csv(input.clean_data()[0]["datapath"]),
                                        osm_path=osm_path,
                                        boundary=boundary_import(
                                        boundary_file=input.feature_boundary()[0]["datapath"]),
                                        pois=list(input.point_features()),
                                        lines=list(input.line_features()),
                                        polygons=list(input.polygon_features()),
                                        point_dist_type=input.point_configuration(),
                                        point_dist=input.point_distance(),
                                        polygon_dist_type=input.polygon_distance(),
                                        polygon_dist=input.polygon_distance(),
                                        saved_path=download_path)

            return render.DataGrid(preparation_feature_data())


    def preparation_feature_data():
        file = input.clean_data()
        path = file[0]["datapath"]
        download_path = os.path.dirname(path) + '_features.csv'
        clean_df = pd.read_csv(path)
        feature_df = pd.read_csv(download_path)
        clean_feature = pd.concat([feature_df, clean_df], axis=1).reset_index(drop=True)
        clean_feature.to_csv(os.path.dirname(path) + '_clean_features.csv', index=False)
        return clean_feature

    @render.ui
    @reactive.event(input.feature_preview)
    def feature_records():
        if input.clean_data() is not None:
            return len(preparation_feature_data())

    @render.ui
    @reactive.event(input.feature_preview)
    def feature_columns():
        if input.clean_data() is not None:
            return len(preparation_feature_data().columns) - 1

    @reactive.effect
    @reactive.event(input.feature_preview)
    def _():
        if input.clean_data() is not None:
            ui.update_selectize(
                'map_features',
                choices=preparation_feature_data().columns.tolist(),
                selected=preparation_feature_data().columns.tolist()[-1])

    @render_widget
    @reactive.event(input.map_show)
    def feature_map():
        if input.clean_data() is not None:
            data = preparation_feature_data()
            fig = px.scatter_mapbox(data,
                                    lat='Latitude',
                                    lon='Longitude',
                                    size=input.map_features(),
                                    color=data.columns.tolist()[-1],
                                    center=dict(lat=data['Latitude'].mean(),
                                                lon=data['Longitude'].mean()),
                                    mapbox_style='carto-positron',
                                    color_continuous_scale=px.colors.sequential.Viridis,
                                    opacity=0.5,
                                    zoom=11)
            return fig

    @render.data_frame
    @reactive.event(input.feature_preview)
    def statistics():
        if input.clean_data() is not None:
            return statistics(preparation_feature_data().iloc[:, 1:])

    @render_widget
    @reactive.event(input.feature_preview)
    def correlation():
        if input.clean_data() is not None:
            data = preparation_feature_data().iloc[:, 1:]
            corr = data.corr(method='pearson')
            fig = px.imshow(np.array(corr), x=data.columns, y=data.columns)
            return fig

    @render.download
    def feature_download():
        file = input.clean_data()
        path = file[0]["datapath"]
        download_path = os.path.dirname(path) + '_clean_features.csv'
        return download_path

    '''
    Model Pipeline Generation & Evaluation
    '''
    # update target column
    @reactive.effect
    def update_target_col():
        if input.final() is not None:
            ui.update_select(
                'target_col',
                choices=pd.read_csv(input.final()[0]["datapath"]).columns.tolist())

    @render.ui
    @reactive.event(input.automl)
    def automl_choice():
        if input.automl() is True:
            return ui.input_selectize(
        "automl_choices",
        "Available AutoML Frameworks:",
        {"tpot": "TPOT",
         "h2o": "H2O",
         # "autogluon": "AutoGluon",
         'pycaret': 'PyCaret',
         'flaml': 'FLAML',
         'mjar':'mjar-supervised'}, selected=None,
        multiple=False,
    ), ui.input_switch("advanced", "Advanced Configuration", False),

    @render.ui
    @reactive.event(input.advanced)
    def automl_configuration():
        if input.advanced() == 1:
            if input.automl_choices() == 'tpot':
                return ui.card(
                    ui.card_header('TPOT Configuration'),
                    ui.input_numeric("tpot_generation", "Number of generations", 5, min=5, max=100, step=5),
                    ui.input_numeric("tpot_population", "Population size", 10, min=10, max=500, step=10),
                    ui.input_slider("tpot_mutation_rate", "Mutation rate", 0, 1, 0.9, step=0.05),
                    ui.input_slider("tpot_crossover_rate", "Crossover rate", 0, 1, 0.1, step=0.05),
                              )

            elif input.automl_choices() == 'h2o':
                return ui.card(
                    ui.card_header('H2O Configuration'),
                    ui.input_numeric("h2o_max_models", "Maximum number of models to build", 10, min=5, max=100, step=5),
                              )

            elif input.automl_choices() == 'autogluon':
                return ui.card(
                    ui.card_header('AutoGluon Configuration'),
                    ui.input_selectize("autogluon_presets", "Maximum number of models to build",
                                       ['best_quality',
                                        'high_quality',
                                        'good_quality',
                                        'medium_quality',
                                        'optimize_for_deployment',
                                        'interpretable',
                                        'ignore_text'], multiple=False)
                              )

            elif input.automl_choices() == 'pycaret':
                return ui.card(
                    ui.card_header('PyCaret Configuration'),
                    ui.input_selectize("pycaret_feature_selection", "Feature selection method",
                                       {'univariate': 'Uses sklearn’s SelectKBest',
                                               'classic': 'Uses sklearn’s SelectFromModel',
                                               'sequential': 'Uses sklearn’s SequentialFeatureSelector'},
                                       selected='Uses sklearn’s SelectFromModel'),
                    ui.input_slider("pycaret_n_features", "Features to be selected", 0.2, 1, 0.2, step=0.05),
                              )

            elif input.automl_choices() == 'flaml':
                return ui.card(
                    ui.card_header('FLAML Configuration'),
                    ui.input_numeric("flaml_time_budget", "Time budget", 60, min=60, max=3600, step=60),
                              )

            elif input.automl_choices() == 'mjar':
                return ui.card(
                    ui.card_header('mjar-supervised Configuration'),
                    ui.input_numeric("mjar_time_budget", "Total time limit", 300, min=600, max=3600, step=60),
                    ui.input_selectize("mjar_mode", "Mode",
                                       {'explain': 'Explain',
                                        'perform': 'Perform',
                                        'compete': 'Compete',
                                        'optuna': 'Optuna'})
                    )

    @render.data_frame
    @reactive.event(input.model_training, ignore_none=False)
    def leaderboard():
        if input.final() is not None:
            path = input.final()[0]["datapath"]
            data = pd.read_csv(path)

            # User AutoML4RPV
            if input.automl() is False:
                path = input.final()[0]["datapath"]
                # download_path = os.path.dirname(path) + '_feature.csv'
                fit(saved_path=os.path.dirname(path),
                    data=pd.read_csv(input.final()[0]["datapath"]),
                    target=input.target_col(),
                    train_size=input.train_test(),
                    feature_selection_config=input.mrmr())
                return pipelines(data=pipeline_info())

            # User open-source AutoML frameworks
            else:
                x_train, x_test, y_train, y_test = train_test_split(data.drop(input.target_col(), axis=1),
                                                                    data[input.target_col()],
                                                                    train_size=input.train_test(),
                                                                    random_state=0)
                train = pd.concat([x_train, y_train], axis=1).reset_index(drop=True)
                test = pd.concat([x_test, y_test], axis=1).reset_index(drop=True)

                # tpot
                if input.automl_choices() == 'tpot':
                    if input.advanced() is True:
                        tpot_results = benchmark_tpot(train=train,
                                                      test=test,
                                                      target=input.target_col(),
                                                      generation=input.tpot_generation(),
                                                      population_size=input.tpot_population(),
                                                      mutation=input.tpot_mutation_rate(),
                                                      crossover=input.tpot_crossover_rate(),
                                                      verbosity=2)
                    else:
                        tpot_results = benchmark_tpot(train=train,
                                                      test=test,
                                                      target=input.target_col())

                    tpot_results = pd.DataFrame([tpot_results])
                    return tpot_results

                # H2O
                if input.automl_choices() == 'h2o':
                    if input.advanced() is True:
                        h2o_results = benchmark_h2o(train=train,
                                                      test=test,
                                                    target=input.target_col(),
                                                    max_models=input.h2o_max_models())
                    else:
                        h2o_results = benchmark_h2o(train=train,
                                                      test=test,
                                                    target=input.target_col())

                    h2o_results = pd.DataFrame([h2o_results])
                    # h2o_leaderboard = h2o_results[1]
                    return h2o_results

                # FLAML
                if input.automl_choices() == 'flaml':
                    if input.advanced() is True:
                        flaml_results = benchmark_flaml(train=train,
                                                      test=test,
                                                    target=input.target_col(),
                                                    time_budget=input.flaml_time_budget())
                    else:
                        flaml_results = benchmark_flaml(train=train,
                                                      test=test,
                                                    target=input.target_col(),
                                                    time_budget=input.flaml_time_budget())
                    flaml_results = pd.DataFrame([flaml_results])
                    return flaml_results

                # PyCaret
                if input.automl_choices() == 'pycaret':
                    if input.advanced() is True:
                        pycaret_results = benchmark_pycaret(train=train,
                                                            test=test,
                                                        target=input.target_col(),
                                                        feature_selection_method=input.pycaret_feature_selection(),
                                                        n_features_to_select=input.pycaret_n_features())
                    else:
                        pycaret_results = benchmark_pycaret(train=train,
                                                            test=test,
                                                        target=input.target_col())
                    pycaret_results = pd.DataFrame([pycaret_results])
                    return pycaret_results

                # mjar
                if input.automl_choices() == 'mjar':
                    if input.advanced() is True:
                       mjar_results = benchmark_mjar(train=train,
                                                            test=test,
                                                     target=input.target_col(),
                                                     total_time_limit=input.mjar_time_budget(),
                                                     mode=input.mjar_mode())
                    else:
                        mjar_results = benchmark_mjar(train=train,
                                                            test=test,
                                                      target=input.target_col())
                    mjar_results = pd.DataFrame([mjar_results])
                    return mjar_results

    def pipeline_info():
        try:
            path = input.final()[0]["datapath"]
            saved_path = os.path.dirname(path)
            pipeline_results = pd.read_csv(saved_path + '\\pipeline_results.csv')
            return pipeline_results

        except FileNotFoundError:
            pipeline_results = pd.DataFrame()
            return pipeline_results


    @render.ui
    @reactive.event(input.model_training, ignore_none=False)
    def pipelines_num():
        if input.final() is not None and input.automl() is False:
            return len(pipeline_info())

    @render.ui
    @reactive.event(input.model_training, ignore_none=False)
    def training_time():
        if input.final() is not None and input.automl() is False:
            time = '%.1f' % (pipeline_info()['Training time'].sum() / 60)
            return '{} min'.format(time)

    @render.ui
    @reactive.event(input.model_training, ignore_none=False)
    def best_pipeline():
        if input.final() is not None and input.automl() is False:
            return best_pipelines(data=pipeline_info()).reset_index().loc[0, 'Pipeline_ID']

    # update selected features
    # @reactive.effect
    # def _():
    #     if input.final() is not None:
    #         choices = pipeline_info()['Pipeline_ID'].tolist()
    #         ui.update_selectize(
    #             'pipeline_selection',
    #             choices=choices, )

    # @render.plot
    # def taylor_plot():
    #     if (input.final() is not None) & (input.pipeline_selection() is not None):
    #         path = input.final()[0]["datapath"]
    #         saved_path = os.path.dirname(path)
    #         fig = taylor_diagram(saved_path=saved_path,
    #                              data=pipeline_info(),
    #                              pipeline_ids=input.pipeline_selection())
    #         return fig

    # @render.plot
    # @reactive.event(input.model_training, ignore_none=False)
    # def regression_plot_test():
    #     if input.final() is not None:
    #         path = input.final()[0]["datapath"]
    #         saved_path = os.path.dirname(path)
    #         fig = reg_plot(saved_path=saved_path,
    #                        data=pd.read_csv(input.final()[0]["datapath"]),
    #                        train_size=input.train_test(),
    #                        target='test')
    #         return fig
    #
    # @render.plot
    # @reactive.event(input.model_training, ignore_none=False)
    # def regression_plot_train():
    #     if input.final() is not None:
    #         path = input.final()[0]["datapath"]
    #         saved_path = os.path.dirname(path)
    #         fig = reg_plot(saved_path=saved_path,
    #                        data=pd.read_csv(input.final()[0]["datapath"]),
    #                        train_size=input.train_test(),
    #                        target='train')
    #         return fig

    '''Explainable Artificial Intelligence'''

    # update target column
    @reactive.effect
    def update_xai_col():
        if input.xai_data() is not None:
            ui.update_select(
                'xai_target_col',
                choices=pd.read_csv(input.xai_data()[0]["datapath"]).columns.tolist())

    # @render.ui
    # @reactive.event(input.pkl)
    # def pkl_upload():
    #     if input.pkl():
    #         return ui.card(
    #             ui.card_header('Model Upload'),
    #             ui.input_file("pkl_upload", "Upload your model", accept=['.pkl'],
    #                           placeholder='No file selected'),
    #         )

    def xai_model():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            data = pd.read_csv(input.xai_data()[0]["datapath"])
            saved_path = os.path.dirname(input.xai_data()[0]["datapath"])
            fi = feature_importance_(x=data.drop(columns=[input.xai_target_col()]),
                                     y=data[input.xai_target_col()],
                                     model=joblib.load(model)[0],
                                     saved_path=saved_path)
            return fi

    # pfi feature importance
    @render.plot
    @reactive.event(input.model_explain)
    def pfi():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            data = pd.read_csv(input.xai_data()[0]["datapath"])
            saved_path = os.path.dirname(input.xai_data()[0]["datapath"])
            fi = feature_importance_(x=data.drop(columns=[input.xai_target_col()]),
                                     y=data[input.xai_target_col()],
                                     model=joblib.load(model)[0],
                                     saved_path=saved_path)
            fi.pfi()
            return xai_model().pfi_figure()

    # shap summary plot
    @render.plot
    @reactive.event(input.model_explain)
    def shap_global():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            data = pd.read_csv(input.xai_data()[0]["datapath"])
            saved_path = os.path.dirname(input.xai_data()[0]["datapath"])
            fi = feature_importance_(x=data.drop(columns=[input.xai_target_col()]),
                                     y=data[input.xai_target_col()],
                                     model=joblib.load(model)[0],
                                     saved_path=saved_path)
            fi.shap_analysis()
            return xai_model().shap_summary(max_display=10)

    # update selected features
    @reactive.effect
    # @reactive.event(input.model_explain)
    def _():
        if input.xai_data() is not None:
            choices = pd.read_csv(input.xai_data()[0]["datapath"]).columns.tolist()
            ui.update_selectize(
                'xai_feature',
                choices=choices, )

    # pdp
    @render.plot
    @reactive.event(input.xai_plot)
    def pdp():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            data = pd.read_csv(input.xai_data()[0]["datapath"])
            saved_path = os.path.dirname(input.xai_data()[0]["datapath"])
            fi = feature_importance_(x=data.drop(columns=[input.xai_target_col()]),
                                     y=data[input.xai_target_col()],
                                     model=joblib.load(model)[0],
                                     saved_path=saved_path)
            return fi.pdp(feature_name=input.xai_feature())

    # shap_dp
    @render.plot
    @reactive.event(input.xai_plot)
    def shap_dp():
        if input.xai_data() is not None and input.pkl_upload() is not None:
            model = input.pkl_upload()[0]["datapath"]
            data = pd.read_csv(input.xai_data()[0]["datapath"])
            saved_path = os.path.dirname(input.xai_data()[0]["datapath"])
            fi = feature_importance_(x=data.drop(columns=[input.xai_target_col()]),
                                     y=data[input.xai_target_col()],
                                     model=joblib.load(model)[0],
                                     saved_path=saved_path)
            return fi.shap_dependence(feature_name=input.xai_feature())

    # spatial map of shap values
    @render_widget
    @reactive.event(input.xai_plot)
    def shap_map():
        if input.xai_feature() is not None:
            if input.xai_data() is not None and input.pkl_upload() is not None:
                model = input.pkl_upload()[0]["datapath"]
                data = pd.read_csv(input.xai_data()[0]["datapath"])
                saved_path = os.path.dirname(input.xai_data()[0]["datapath"])
                fi = feature_importance_(x=data.drop(columns=[input.xai_target_col()]),
                                         y=data[input.xai_target_col()],
                                         model=joblib.load(model)[0],
                                         saved_path=saved_path)
                return fi.shap_spatial(selected_feature=input.xai_feature())

app = App(app_ui, server)
app.run()