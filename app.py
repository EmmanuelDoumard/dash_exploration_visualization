"""Instantiate a Dash app."""
import numpy as np
import pandas as pd
import dash
import dash_table
import dash_html_components as html
import dash_core_components as dcc
from dash.dependencies import Input, Output, State
from anndata import AnnData,read_h5ad
import math
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib
from io import BytesIO
import base64
import shap
import time
import pickle5
import json
from dash.exceptions import PreventUpdate


adata = read_h5ad("data/adataShap.h5ad")

def getData(adata,ind,obsY='age'):
    adata = adata[:,ind]
    adata = adata[adata.to_df().dropna(thresh=math.floor(0.95*sum(ind))).index,:]
    data = adata.to_df()
    return data, adata.obs[obsY], adata.var.loc[:,'SAS']

def plot_shap_context_heatmap(bst,adata):
    X = adata.to_df()
    y = adata.obs["age"]
    delta = 1
    imp=[]
    base_values = []
    shap_values_tab = []
    explainers = []
    age_range = range(int(y.min()),int(y.max())+1)
    for age in age_range:
        age_min = age-delta
        age_max = age+delta
        cond = (y>=age_min) & (y<=age_max)
        explainer = shap.TreeExplainer(bst,X[cond])
        shap_values = explainer.shap_values(X[y==age],check_additivity=False)
        
        base_values.append(explainer.expected_value)
        explainers.append(explainer)
        shap_values_tab.append(shap_values)
        imp.append(np.abs(shap_values).mean(axis=0))
        
    df_imp = pd.DataFrame(imp,columns=adata.var["SAS"].values,index=[str(age) for age in age_range])

    plt.figure(figsize=(15,15))
    #sns.heatmap(df_imp[df_imp.mean().sort_values(ascending=False).index].T,cmap="viridis")
    out_img = BytesIO()
    plt.savefig(out_img, format='jpg',bbox_inches='tight')
    plt.close('all')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/jpg;base64,{}".format(encoded)

def to_dropdown(item_list):
    dropdown_list = []
    for item in item_list:
        dropdown_list.append({'label': item, 'value': item})
    return dropdown_list

def get_scatter(df, x, y, selectedpoints, width=800, height=700):
    fig = px.scatter(df, x=x, y=y, width=width, height=height, labels={"x":"Real age","y":"Predicted age"},title="Predictions scatterplot")

    fig.update_traces(selectedpoints=selectedpoints,
                      #customdata=df.index
                      mode='markers', unselected={'marker': { 'color':'grey','opacity': 0.1}}
                      )

    x_space = np.linspace(x.min(),x.max(),int(x.max()-x.min()+1))
    fig.add_trace(go.Scatter(x=x_space,y=x_space,mode="lines"))
    fig.update_layout(dragmode='lasso')

    return fig

def get_umap(df, x, y, color, selectedpoints, width=800, height=700):
    df["size"] = 2*np.ones(df.shape[0])
    fig = px.scatter(df,x=x,y=y,color=color,color_continuous_scale='spectral_r',
                    size="size",size_max=4,width=800,height=700,
                    labels={"umap1":"UMAP first dimension","umap2":"UMAP second dimension"},
                    title="UMAP of SHAP values",
                    hover_data={"Age":True,color:True,"umap1":False,"umap2":False,"size":False})

    fig.update_traces(selectedpoints=selectedpoints,
                    #customdata=df.index
                    mode='markers', unselected={'marker': { 'color':'grey','opacity': 0.1}}
                    )
    fig.update_layout(dragmode='lasso')

    return fig

def multipleSelection(selection1,selection2):
    if (selection1 is None) and (selection2 is None):
        return None
    if selection1 and selection1["points"]:
        indexes1 = [point["pointIndex"] for point in selection1["points"]]
    else:
        indexes1 = [i for i in range(adata.to_df().shape[0])]
    if selection2 and selection2["points"]:
        indexes2 = [point["pointIndex"] for point in selection2["points"]]
    else:
        indexes2 = [i for i in range(adata.to_df().shape[0])]
    indexes = np.intersect1d(indexes1,indexes2)
    return indexes

def arrays_equal(array1,array2):
    if (array1 is None) or (array2 is None):
        return False
    elif len(array1) == len(array2):
        return (array1==array2).all()
    else:
        return False

def create_data_table(df):
    """Create Dash datatable from Pandas DataFrame."""
    table = dash_table.DataTable(
        id='database-table',
        columns=[{"name": i, "id": i} for i in df.columns],
        data=df.to_dict('records'),
        sort_action="native",
        sort_mode='native',
        page_size=300
    )
    return table

def figSaveWebFormat():
    plt.tight_layout()
    out_img = BytesIO()
    plt.savefig(out_img, format='jpg')
    out_img.seek(0)  # rewind file
    encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    return "data:image/jpg;base64,{}".format(encoded)

def init_dashboard():
    """Create a Plotly Dash dashboard."""
    app = dash.Dash(__name__)
    #server = app.server

    fig = get_scatter(adata.obs, x=adata.obs['age'], y=adata.obs['pred_age'], selectedpoints=adata.obs.index)

    umap_df = pd.DataFrame({"umap1":adata.obsm['umap_shapValues'][:,0],"umap2":adata.obsm['umap_shapValues'][:,1],"Age":adata.obs['age']},index=adata.obs.index)
    fig2 = get_umap(umap_df,x="umap1",y="umap2",color="Age",selectedpoints=umap_df.index)

    # divPareto = html.Div(children=[
    #                         dcc.Graph(id='pareto-plot',figure=pareto_fig),
    #                         html.Div([dcc.Dropdown(id='dropdown-pareto-vars',options=[], placeholder="Select a single point to show the variables")]),
    #                         html.Div(id='div-hist-plot')
    #                     ],
    #                     id='div-pareto')
    div1 = html.Div(
                    children=[
                        #html.Div([html.H3(children=["TEST"])]),
                        dcc.Graph(id='scatter-plot',figure=fig,style={'width':'49%','display':'inline-block'}),
                        html.Div([dcc.Dropdown(id='dropdown-umap-type',options=[
                                                                                {'label':'Data','value':'umap_data'},
                                                                                {'label':'Data standard scaled','value':'umap_data_standardScaler'},
                                                                                {'label':'SHAP values','value':'umap_shapValues'},
                                                                                {'label':'SHAP values contextualized','value':'umap_shapValuesContextualized'}
                                                                                ],
                                                value="umap_shapValues",
                                                placeholder="UMAP type"
                                                ),
                                    dcc.Dropdown(id='dropdown-umap-color',options=[{'label':'Age','value':'Age'}]+
                                                                                [{'label':k, 'value':k} for k in adata.var["SAS"].values],
                                                value="Age",
                                                placeholder="UMAP variable color"),
                                    dcc.Graph(id='umap-shap-values',figure=fig2)
                                    ],
                                style={'width':'49%','display':'inline-block'}),
                        dcc.Loading([
                            html.Div([html.Button('Update Decision Plot', id='updateDecision-button')]),
                            html.Div([html.Img(id = 'selected-data', src = '')],style={'width':'49%','display':'inline-block'}),
                            html.Div([html.Img(id = 'selected-data-mean', src = '')],style={'width':'49%','display':'inline-block'}),
                            html.Div([dcc.Dropdown(id='dropdown_age',options=[],placeholder="Select an age amongst selected individuals")]),
                        ]),
                        dcc.Loading([
                            html.Div([html.Img(id = 'contextualized-shap', src = '')])
                        ])#,style={'width':'49%','display':'inline-block'})
                    ],
                    id='dash-container'
                )

    div2 = html.Div(
                    children=[
                        html.Div([html.Img(id = 'heatmap-context', src='')]),#plot_shap_context_heatmap(bst,adata))],style={'width':'49%','display':'inline-block'}),
                        html.Div([dcc.Dropdown(id='dropdown-age-complet',options=to_dropdown(range(int(adata.obs["age"].min()),int(adata.obs["age"].max())+1)))]),
                        html.Div([html.Img(id = 'contextualized-summary-shap', src = '')]),
                        html.Div([dcc.Dropdown(id = 'dropdown-vars', options=[],placeholder="Dependence plot variable")]),
                        html.Div([dcc.Dropdown(id = 'dropdown-vars-interaction', options=[], placeholder="Interaction variable")]),
                        html.Div([dcc.RangeSlider(id = 'xrangeslider')],id='xrangeslider-container'),
                        html.Div([html.Img(id = 'dependence-plot', src = '')])
                    ]
    )
    app.layout = html.Div([
        dcc.Tabs(id="tabs", value='tab-1', children=[
            #dcc.Tab(label='Sélection et visualisaton du dataset', value = 'tab-0', children=divPareto),
            dcc.Tab(label='Scatter plot des prédictions', value='tab-1', children=div1),
            dcc.Tab(label='Exploration des variables contextualisées', value='tab-2', children=div2)
        ]),
        dcc.Store(id="selection-indexes-stored"),
        dcc.Store(id="context-age")
    ])

    init_callbacks(app)

    return app

def init_callbacks(app):
    # @app.callback(
    #     Output('dropdown-pareto-vars','options'),
    #     Output('dropdown-pareto-vars','placeholder'),
    #     Input('pareto-plot','selectedData'))
    # def fill_dropdown_pareto_vars(selectedPareto):
    #     if selectedPareto and selectedPareto["points"]:
    #         indexes = [point["pointIndex"] for point in selectedPareto["points"]]
    #     else:
    #         return [],dash.no_update
    #     if len(indexes)==1:
    #         X, y, fe = getData(adata, pareto1.iloc[indexes[0]].loc["ind"])
    #         dropdown_vars = []
    #         for col in X.columns:
    #             dropdown_vars.append({'label':adata.var.loc[col,'SAS'],'value':adata.var.loc[col,'SAS']})
    #         return dropdown_vars,"Select a variable"
    #     else:
    #         return [],"More than a single point has been selected"

    # @app.callback(
    #     Output('div-hist-plot','children'),
    #     Input('dropdown-pareto-vars','value'),
    #     Input('pareto-plot','selectedData'))
    # def plot_var_hist(var,selectedPareto):
    #     if selectedPareto and selectedPareto["points"]:
    #         indexes = [point["pointIndex"] for point in selectedPareto["points"]]
    #     else:
    #         return None
    #     if len(indexes)==1 and var:
    #         print("Coucou")
    #         X, y, fe = getData(adata, pareto1.iloc[indexes[0]].loc["ind"])
    #         fig = px.scatter(x=X.loc[:,adata.var[adata.var["SAS"]==var].index[0]],
    #                             y=y,
    #                             labels={"x":var,"y":"Age"},
    #                             title="Joint plot for {}. NaNs proportion : {}".format(var,X.loc[:,adata.var[adata.var["SAS"]==var].index[0]].isna().sum()/X.shape[0]),
    #                             marginal_x="histogram",
    #                             marginal_y="histogram",
    #                             width=1200,
    #                             height=1200)
    #         return dcc.Graph(figure=fig)
    #     else:
    #         return None
    @app.callback(
        Output('umap-shap-values','figure'),
        Input('dropdown-umap-type','value'),
        Input('dropdown-umap-color','value'),
        Input('selection-indexes-stored','data'))
    def plot_umap(umap_type,umap_color,indexes):
        dataframe_dict = {"umap1":adata.obsm[umap_type][:,0],
                          "umap2":adata.obsm[umap_type][:,1],
                          "Age":adata.obs["age"]}
        if umap_color!="Age":
            dataframe_dict[umap_color] = adata.to_df().loc[:,adata.var[adata.var["SAS"]==umap_color].index[0]]
        umap_df = pd.DataFrame(dataframe_dict,
                               index=adata.obs.index)
        return get_umap(umap_df, x="umap1", y="umap2", color=umap_color, selectedpoints=indexes)

    @app.callback(
        Output('selection-indexes-stored','data'),
        Input('umap-shap-values','selectedData'),
        Input('scatter-plot', 'selectedData'))
    def store_indexes_selections(selectionScatter,selectionUmap):
        indexes = multipleSelection(selectionScatter,selectionUmap)
        if (indexes is not None):
            return indexes
        else:
            return None

    @app.callback(
        Output('scatter-plot','figure'),
        Input('selection-indexes-stored','data'))
    def plot_scatter(indexes):
        return get_scatter(adata.obs, x=adata.obs['age'], y=adata.obs['pred_age'], selectedpoints=indexes)

    @app.callback(
        Output('contextualized-shap', 'src'),
        Input('dropdown_age', 'value'),
        Input('selection-indexes-stored','data'))
    def plot_contextualized_shap(age,indexes):
        ctx = dash.callback_context
        if ctx.triggered[0]["prop_id"] == "selection-indexes-stored.data": #Trigger source is indexes
            return ''   #Cancel figure
        elif (age is not None):
            plt.figure()
            adataAge = adata[adata.obs["age"]==int(age),:]
            adataAge = adataAge[adataAge.obs.index.isin(adata.obs.index[indexes])]
            shap.decision_plot(adata.obs.loc[adata.obs["age"]==int(age),'pred_age'].mean(), 
                                adataAge.layers['shapValuesContextualized'],
                                adataAge.X,
                                feature_names=adata.var["SAS"].to_list(),
                                show=False,
                                ignore_warnings=True)
            plt.title("SHAP values contextualisées")

            img3 = figSaveWebFormat()
            return img3
        raise PreventUpdate

    # @app.callback(
    #     Output('contextualized-shap', 'src'),
    #     Input('selection-indexes-stored','data'),
    #     State('dropdown_age', 'value'))
    # def plot_contextualized_shap(value,selectionIndexes):
    #     return ''

    @app.callback(
        Output('selected-data', 'src'),
        Output('selected-data-mean','src'),
        Output('dropdown_age', 'options'),
        Input('selection-indexes-stored','data'),
        Input('updateDecision-button','n_clicks'))
    def display_selected_data(indexes,osef):
        if dash.callback_context.triggered[0]["prop_id"] != "updateDecision-button.n_clicks": #Trigger source is not the button
            return '','',[]
        if (indexes is not None) and (len(indexes)!=0):
            plt.figure()
            #plt.scatter(df.iloc[indexes].loc[:,"LBXCOT"],df.iloc[indexes].loc[:,"BMXBMI"])
            shap.decision_plot(adata.obs['pred_age'].mean(), 
                                adata.layers['shapValues'][indexes,:],
                                adata.to_df().iloc[indexes],
                                feature_names=adata.var["SAS"].to_list(),
                                show=False,
                                ignore_warnings=True)
            plt.title("SHAP values de la sélection")
            #plt.savefig("figures/test.jpg")

            
            img1 = figSaveWebFormat()

            plt.figure()
            #plt.scatter(df.iloc[indexes].loc[:,"LBXCOT"],df.iloc[indexes].loc[:,"BMXBMI"])
            shap.decision_plot(adata.obs['pred_age'].mean(), 
                                adata.layers['shapValues'][indexes,:].mean(axis=0),
                                adata.to_df().iloc[indexes].mean(axis=0),
                                feature_names=adata.var["SAS"].to_list(),
                                show=False,
                                ignore_warnings=True)
            plt.title("Moyenne des SHAP values de la sélection")
            #plt.savefig("figures/test.jpg")

            img2 = figSaveWebFormat()

            dropdown_age = [{'label': str(int(unique_age)), 'value': str(int(unique_age))} for unique_age in np.sort(adata.obs["age"].iloc[indexes].unique())]
            return img1,img2, dropdown_age
        raise PreventUpdate

    @app.callback(
        Output('dropdown-vars','options'),
        Output('dropdown-vars-interaction','options'),
        Input('dropdown-age-complet','value'))
    def sort_list_dropdown_vars(age):
        if age is not None:
            dropdown_vars=[]
            shap_values_context = adata.layers['shapValuesContextualized'][adata.obs['age']==int(age),:]
            df_shap_values = pd.DataFrame(shap_values_context,columns=adata.var["SAS"])
            for col in df_shap_values.abs().mean().sort_values(ascending=False).index:
                dropdown_vars.append({'label': col, 'value': col})
            dropdown_vars_interaction = dropdown_vars.copy()
            return dropdown_vars, dropdown_vars_interaction
        raise PreventUpdate

    @app.callback(
        Output('xrangeslider-container', 'children'),
        Input('dropdown-vars', 'value'),
        Input('dropdown-age-complet', 'value'))
    def generate_slider(var,age):
        if (age is not None) and (var is not None):
            X_context = adata.X[adata.obs['age']==int(age),:]
            i=np.where(adata.var['SAS']==var)[0][0]
            xmin = np.nanmin(adata.X[:,i])
            xmax = np.nanmax(adata.X[:,i])
            xrange = [xmin, xmax]
            marks_values = np.linspace(xmin, xmax, 10)
            rangeslider_div = [dcc.RangeSlider(id='xrangeslider',
                                    min=xmin,
                                    max=xmax,
                                    value=[xrange[0], xrange[1]],
                                    step=0.01,
                                    marks={int(m) if m % 1 == 0 else m:"%.2f"%m for m in marks_values},
                                    tooltip={"always_visible":True, "placement":'bottom'})]
            return rangeslider_div
        raise PreventUpdate

    @app.callback(
        Output('contextualized-summary-shap', 'src'),
        Input('dropdown-age-complet', 'value'))
    def plot_contextualized_summary_shap(age):
        if age is not None:
            shap_values_context = adata.layers['shapValuesContextualized'][adata.obs['age']==int(age),:]
            X_context = adata.X[adata.obs['age']==int(age),:]
            plt.figure()
            shap.summary_plot(shap_values_context,
                              X_context,
                              feature_names=adata.var["SAS"].to_list(),
                              show=False)
            plt.title("SHAP values contextualisées pour les individus de {} ans".format(age))

            img1 = figSaveWebFormat()
            return img1
        raise PreventUpdate

    @app.callback(
        Output('dependence-plot','src'),
        Input('dropdown-vars-interaction','value'),
        Input('xrangeslider','value'),
        State('dropdown-age-complet', 'value'),
        State('dropdown-vars', 'value'))
    def plot_contextualized_dependence_shap(var_interaction,xrange,age,var):
        if (age is not None) and (var is not None):
            shap_values_context = adata.layers['shapValuesContextualized'][adata.obs['age']==int(age),:]
            X_context = adata.X[adata.obs['age']==int(age),:]
            fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(20,8))
            shap.dependence_plot(var,
                                shap_values_context,
                                X_context,
                                interaction_index=var_interaction,
                                feature_names=adata.var["SAS"].to_list(),
                                show=False,
                                ax=ax1,
                                xmin=xrange[0],
                                xmax=xrange[1])
            ax1.set_title("Dependence plot contextualisé pour les individus de {} ans".format(age))
            
            shap.dependence_plot(var,
                                adata.layers['shapValues'],
                                adata.X,
                                interaction_index=var_interaction,
                                feature_names=adata.var["SAS"].to_list(),
                                show=False,
                                ax=ax2,
                                xmin=xrange[0],
                                xmax=xrange[1])
            ax2.set_title("Dependence plot global")

            print("Saving image")
            img = figSaveWebFormat()
            return img
        else:
            return ''


    # @app.callback(
    #     Output('contextualized-summary-shap', 'src'),
    #     Output('dependence-plot', 'src'),
    #     Output('dropdown-vars', 'options'),
    #     Output('dropdown-vars-interaction', 'options'),
    #     Output('xrangeslider-container', 'children'),
    #     Input('dropdown-vars', 'value'),
    #     Input('dropdown-age-complet', 'value'),
    #     Input('dropdown-vars-interaction','value'),
    #     Input('xrangeslider','value'))
    # def plot_contextualized_dependence_shap(var,age,var_interaction,xrange):
    #     global age_context
    #     global previous_var
    #     print("Age : ",age)
    #     print("Age context : ",age_context)
    #     print("Var : ",var)
    #     print("Var interaction :", var_interaction)
    #     print("xrange : ",xrange)
    #     img1, img2 = None, None
    #     xmin, xmax = None, None
    #     dropdown_vars = []
    #     dropdown_vars_interaction = []
    #     rangeslider_div = [dcc.RangeSlider(id = 'xrangeslider')]
    #     if xrange is None:
    #         xrange=[None,None]
    #     if age is not None:
    #         if age == age_context: #Did not change age
    #             img1 = dash.no_update
    #             dropdown_vars = dash.no_update
    #             dropdown_vars_interaction = dash.no_update
    #         else:
    #             print("Hello sum 1")
    #             age_context = age
    #             explainer_context = shap.TreeExplainer(bst,df.loc[adata.obs["age"]==int(age)])
    #             shap_values_context = explainer_context.shap_values(df.loc[adata.obs["age"]==int(age)])

    #             plt.figure()
    #             shap.summary_plot(shap_values_context,
    #                             df.loc[adata.obs["age"]==int(age)],
    #                             feature_names=adata.var.loc[df.columns,"SAS"].to_list(),
    #                             show=False)
    #             plt.title("SHAP values contextualisées pour les individus de {} ans".format(age))

    #             out_img = BytesIO()
    #             plt.savefig(out_img, format='jpg',bbox_inches='tight')
    #             plt.close('all')
    #             out_img.seek(0)  # rewind file
    #             encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    #             print("Hello sum 2")
    #             img1 = "data:image/jpg;base64,{}".format(encoded)

    #             df_shap_values = pd.DataFrame(shap_values_context,columns=adata.var.loc[df.columns,"SAS"])
    #             for col in df_shap_values.abs().mean().sort_values(ascending=False).index:
    #                 dropdown_vars.append({'label': col, 'value': col})
    #             dropdown_vars_interaction = dropdown_vars.copy()
    #         if var is not None:
    #             xmin = df.loc[:,adata.var[adata.var["SAS"]==var].index[0]].min()
    #             xmax = df.loc[:,adata.var[adata.var["SAS"]==var].index[0]].max()
    #             if var != previous_var:
    #                 xrange = [xmin, xmax]
    #                 previous_var = var
    #             print("Hello dep 1")

    #             fig, (ax1, ax2) = plt.subplots(1,2,sharey=True,figsize=(20,8))
    #             shap.dependence_plot(var,
    #                                 shap_values_context,
    #                                 df.loc[adata.obs["age"]==int(age)],
    #                                 interaction_index=var_interaction,
    #                                 feature_names=adata.var.loc[df.columns,"SAS"].to_list(),
    #                                 show=False,
    #                                 ax=ax1,
    #                                 xmin=xrange[0],
    #                                 xmax=xrange[1])
    #             ax1.set_title("Dependence plot contextualisé pour les individus de {} ans".format(age))
                
    #             shap.dependence_plot(var,
    #                                 shap_values,
    #                                 df,
    #                                 interaction_index=var_interaction,
    #                                 feature_names=adata.var.loc[df.columns,"SAS"].to_list(),
    #                                 show=False,
    #                                 ax=ax2,
    #                                 xmin=xrange[0],
    #                                 xmax=xrange[1])
    #             ax2.set_title("Dependance plot global")

    #             out_img = BytesIO()
    #             fig.savefig(out_img, format='jpg',bbox_inches='tight')
    #             plt.close('all')
    #             out_img.seek(0)  # rewind file
    #             encoded = base64.b64encode(out_img.read()).decode("ascii").replace("\n", "")
    #             img2 = "data:image/jpg;base64,{}".format(encoded)
    #             marks_values = np.linspace(xmin, xmax, 10)
    #             rangeslider_div = [dcc.RangeSlider(id='xrangeslider',
    #                                                 min=xmin,
    #                                                 max=xmax,
    #                                                 value=[xrange[0], xrange[1]],
    #                                                 step=0.01,
    #                                                 marks={int(m) if m % 1 == 0 else m:"%.2f"%m for m in marks_values},
    #                                                 tooltip={"always_visible":True, "placement":'bottom'})]
        
    #     return img1, img2, dropdown_vars, dropdown_vars_interaction, rangeslider_div

if __name__ == '__main__':
    print("Initialisation du dashboard...")
    app = init_dashboard()
    print("Dashboard initialisé, lancement du serveur")
    app.run_server(host='0.0.0.0', port=8049)