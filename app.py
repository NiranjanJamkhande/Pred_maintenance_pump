import streamlit as st
import xgboost as xgb
import numpy as np
import pandas as pd
import plotly.graph_objs as go
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.preprocessing import MinMaxScaler
from plotly.subplots import make_subplots

# Load your XGBoost model

loaded_model = xgb.Booster()
loaded_model.load_model('xgboost_model.model')

# Define custom background color and text color

CUSTOM_BG_COLOR = '#f0f2f6'  # Light gray background
CUSTOM_TEXT_COLOR = '#333333'  # Dark gray text

def descriptive_tab():

    st.title('Descriptive Analysis')
    st.sidebar.title('Upload CSV File')
    descriptive_file = st.sidebar.file_uploader("Upload a CSV file", type=['csv'])

    if descriptive_file is not None:
        df_plot = pd.read_csv(descriptive_file)

 # Apply SelectKBest class to extract the best 10 features - Univariate feature selection
        x = df_plot.drop(['machine_status', 'Date'], axis=1)
        y = df_plot['machine_status']
        scaler = MinMaxScaler()
        x_scaled = scaler.fit_transform(x)
        bestfeatures = SelectKBest(score_func=chi2, k=4)
        fit = bestfeatures.fit(x_scaled, y)
        dfscores = pd.DataFrame(fit.scores_)
        dfcolumns = pd.DataFrame(x.columns)
        featureScores = pd.concat([dfcolumns, dfscores], axis=1)
        featureScores.columns = ['Feature', 'Score']
        sorted_feature_scores = featureScores.sort_values(by='Score', ascending=False)

        # Define a color scale ('Viridis' color scale)
        color_scale = sorted_feature_scores['Score'][:9]

        # Create a Plotly bar plot
        fig = make_subplots(rows=1, cols=1)

        # Add a horizontal bar trace with color scale
        fig.add_trace(go.Bar(
            x=sorted_feature_scores['Score'][:9],
            y=sorted_feature_scores['Feature'][:9],
            orientation='h',
            marker=dict(color=color_scale, colorscale='Viridis'),  # Use 'Viridis' color scale
            hoverinfo='none'  # Disable hover info
        ))

        # Add annotations for feature importance values and sensor names
        for i, (score, feature) in enumerate(zip(sorted_feature_scores['Score'][:9], sorted_feature_scores['Feature'][:9])):
            fig.add_annotation(
                x=score / 2,
                y=feature,
                text=f"<b>{feature}</b><br><b>Score:</b> <b>{score:.2f}</b>",
                showarrow=False,
                font=dict(size=10, color='black', family='Arial, bold'),  # Adjust font size, color, and make it bold
                xanchor='center',  # Center the text horizontally
                yanchor='middle'   # Center the text vertically
            )

        # Update layout for title and subtitle
        fig.update_layout(
            title={
                'text': "<b>Top 9 Important Features</b>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top',
                'font': {'size': 16, 'color': 'black', 'family': 'Arial, bold'}  # Adjust font size, color, and make it bold
            },
            xaxis=dict(title='<b>Feature Importance Score</b>'),
            yaxis=dict(title='<b>Features</b>', autorange="reversed"),
            showlegend=False  # Hide legend as it's not needed for this plot
        )

        # Add figure title
        fig.update_layout(
            title={
                'text': "<b>Top Nine Important Features</b>",
                'y': 0.95,
                'x': 0.5,
                'xanchor': 'center',
                'yanchor': 'top'
            }
        )


        # Convert the Date column to datetime format
        df_plot['Date'] = pd.to_datetime(df_plot['Date'])

        broken_timestamps = df_plot[df_plot['machine_status'] == 'BROKEN']['Date']

        # Calculate recovery times
        recovery_times = []
        broken_periods = []
        broken_start = None

        for i, row in df_plot.iterrows():
            if row['machine_status'] == 'BROKEN' and broken_start is None:
                broken_start = row['Date']
            elif row['machine_status'] != 'BROKEN' and broken_start is not None:
                broken_end = row['Date']
                broken_periods.append((broken_start, broken_end))
                broken_start = None

        # If the file ends while still broken
        if broken_start is not None:
            broken_periods.append((broken_start, df_plot.iloc[-1]['Date']))

        for start, end in broken_periods:
            recovery_start_rows = df_plot[(df_plot['Date'] > start) & (df_plot['machine_status'] == 'RECOVERING')]
            if not recovery_start_rows.empty:
                recovery_start = recovery_start_rows.iloc[0]['Date']
                recovery_end_rows = df_plot[(df_plot['Date'] > recovery_start) & (df_plot['machine_status'] == 'NORMAL')]
                if not recovery_end_rows.empty:
                    recovery_end = recovery_end_rows.iloc[0]['Date']
                    recovery_time = round((recovery_end - recovery_start).total_seconds() / 3600, 2)
                    recovery_month = recovery_start.strftime('%B %Y')
                    recovery_times.append((recovery_time, recovery_month))

        # Create DataFrame for recovery times
        recovery_df = pd.DataFrame({
            'recovery_period': range(1, len(recovery_times) + 1),
            'recovery_time_hours': [time for time, Month in recovery_times],
            'Month': [Month for time, Month in recovery_times]
        })
        color_scale = px.colors.sequential.Viridis  # You can choose any color scale you prefer
        max_time = recovery_df['recovery_time_hours'].max()
        min_time = recovery_df['recovery_time_hours'].min()

        # Normalize the recovery_time_hours to range [0, 1] for color assignment
        normalized_times = (recovery_df['recovery_time_hours'] - min_time) / (max_time - min_time)

        # Since you want all months to have the same color, we'll use a single color for all months
        # Pick any color from the color scale
        single_color = color_scale[0]

        # Add a text column to include both recovery time and month
        recovery_df['text'] = recovery_df['recovery_time_hours'].apply(lambda x: f"{x:.2f}") +'hours<br>'+recovery_df['Month']

        # Create the bar plot for recovery times
        fig_recovery_times = px.bar(recovery_df, x='recovery_period', y='recovery_time_hours',
                                    title='Recovery Times for Each Failure',
                                    labels={'recovery_time_hours': '<b>Recovery Time (Hours)</b>', 'recovery_period': '<b>Recovery Period</b>'},
                                    hover_data={'recovery_period': True, 'recovery_time_hours': True, 'Month': True},
                                    color='recovery_time_hours',  # Color based on recovery time
                                    color_continuous_scale='Viridis',  # Color scale
                                    color_continuous_midpoint=recovery_df['recovery_time_hours'].median()
                                    )

        fig_recovery_times.update_layout(
            title=dict(
                text='<b>Recovery Times for Each Failure</b>',  # Making title text bold
                font=dict(size=22, family='Arial, sans-serif', color='black'),
                x=0.5,  # Set x position to 0.5 to center horizontally
                xanchor='center',  # Anchor the title text to the center horizontally
                y=0.9,  # Adjust the vertical position if needed
                yanchor='top'  # Anchor the title text to the top of the plot
            )
        )



        # Add annotations for each bar (showing both recovery time and month inside the bar)
        annotations = []
        for index, row in recovery_df.iterrows():
            annotations.append(
                dict(
                    x=row['recovery_period'],
                    y=row['recovery_time_hours'] / 2,  # Placing the text at the center of each bar
                    text=f"<b>{row['recovery_time_hours']:.2f} hours<br>{'<b>' + row['Month'] + '</b>'}</b>",
                    showarrow=False,
                    font=dict(size=12, color='black', family='Arial, bold'),  # Adjust font size and color
                    xanchor='center',  # Center the text horizontally
                    yanchor='middle'   # Center the text vertically
                )
            )

        fig_recovery_times.update_layout(annotations=annotations)

        # Initialize a column for cumulative values


        df_plot['Date'] = pd.to_datetime(df_plot['Date'])

        df_plot['cumulative_value'] = 0

        # Loop through the DataFrame to calculate cumulative values
        cumulative_value = 0
        for i in range(1, len(df_plot)):
            if df_plot.loc[i, 'machine_status'] == 'NORMAL':
                cumulative_value += 1
            df_plot.loc[i, 'cumulative_value'] = cumulative_value

        # Plotting machine status over time
        fig_machine_status = go.Figure()

        # Define color and marker mapping for machine status
        color_map = {'NORMAL': 'blue', 'RECOVERING': 'black'}
        marker_map = {'BROKEN': 'x'}

        # Function to add a trace for a segment
        def add_trace(fig, df_segment, status):
            if status == 'BROKEN':
                fig.add_trace(go.Scatter(
                    x=df_segment['Date'],
                    y=df_segment['cumulative_value'],
                    mode='markers',
                    name=status,
                    marker=dict(color='red', symbol='x', size=10),
                    hovertemplate='Date: %{x}<br>Status: %{text}<br>Cumulative Value: %{y}',
                    text=df_segment['machine_status'],
                    showlegend=False
                ))
            else:
                fig.add_trace(go.Scatter(
                    x=df_segment['Date'],
                    y=df_segment['cumulative_value'],
                    mode='lines',
                    name=status,
                    line=dict(color=color_map[status]),
                    hovertemplate='Date: %{x}<br>Status: %{text}<br>Cumulative Value: %{y}',
                    text=df_segment['machine_status'],
                    showlegend=False
                ))

        # Find segments where machine_status changes
        current_status = df_plot.iloc[0]['machine_status']
        start_idx = 0

        for i in range(1, len(df_plot)):
            if df_plot.iloc[i]['machine_status'] != current_status:
                # Add the trace for the current segment
                add_trace(fig_machine_status, df_plot.iloc[start_idx:i+1], current_status)

                # Update for the new segment
                current_status = df_plot.iloc[i]['machine_status']
                start_idx = i

        # Add the last segment
        add_trace(fig_machine_status, df_plot.iloc[start_idx:], current_status)

        # Update layout with bold titles
        fig_machine_status.update_layout(
            xaxis=dict(
                rangeslider=dict(visible=True),
                type='date',
                title=dict(text='DATE', font=dict(size=18, family='Arial, sans-serif', color='black'))
            ),
            yaxis=dict(
                title=dict(text='PRODUCTION', font=dict(size=18, family='Arial, sans-serif', color='black'))
            ),
            title=dict(
                text='<b>Production Plot of Machine Status Over Time</b>',  # Making title text bold
                font=dict(size=22, family='Arial, sans-serif', color='black'),
                y=0.9,
                x=0.5,  # Set x position to 0.5 to center horizontally
                xanchor='center',  # Anchor the title text to the center horizontally
                yanchor='top'
            ),
            height=800  # Increase the height of the plot
        )



        # Add zoom functionality
        fig_machine_status.update_layout(hovermode='x unified')

        # Remove legend
        fig_machine_status.update_layout(showlegend=False)



        # Render the plots
        st.subheader("Recovery Times for Each Failure")
        st.plotly_chart(fig_recovery_times)

        st.subheader("Production Plot of Machine Status Over Time")
        st.plotly_chart(fig_machine_status)


        st.subheader("Important Features")
        st.plotly_chart(fig)


        # Plotting the sensor data
        fig_sensor_data = go.Figure()

        df_plot.set_index('Date', inplace=True)
        broken = df_plot[df_plot['machine_status'] == 'BROKEN']
        recovering = df_plot[df_plot['machine_status'] == 'RECOVERING']

        # Extract the names of the numerical columns
        df2 = df_plot.drop(columns = ['machine_status', 'cumulative_value'], axis=1)
        
        names = df2.columns

        # Dropdown for selecting a sensor
        selected_sensor = st.selectbox('Select Sensor', names)

        # Add the original sensor data
        fig_sensor_data.add_trace(go.Scatter(x=df_plot.index, y=df_plot[selected_sensor], mode='lines', name='Original',                           line=dict(color='blue')))

        # Add the recovering points
        fig_sensor_data.add_trace(go.Scatter(x=recovering.index, y=recovering[selected_sensor], mode='markers', name='RECOVERING', marker=dict(color='black', size=12, symbol='x')))

        # Add the broken points
        fig_sensor_data.add_trace(go.Scatter(x=broken.index, y=broken[selected_sensor], mode='markers', name='BROKEN', marker=dict(color='red', size=16, symbol='x')))

        # Update layout
        fig_sensor_data.update_layout(title=selected_sensor, xaxis_title='Date', yaxis_title=selected_sensor)



        st.subheader("Sensor Data Plots")
        st.plotly_chart(fig_sensor_data)

def generate_plotly_html(df, broken_actual, broken_predicted, sensor_name):

    fig = go.Figure()
    # Add original data as a line plot
    fig.add_trace(go.Scatter(x=df.index, y=df[sensor_name], mode='lines', name='Original', line=dict(color='blue')))
    # Add markers for broken actual readings
    fig.add_trace(go.Scatter(x=broken_actual.index, y=broken_actual[sensor_name], mode='markers', name='Broken Actual',
                             marker=dict(color='black', size=10, symbol='cross'), hoverinfo='x+y+text',
                             text='Actual Broken: ' + broken_actual.index.strftime('%Y-%m-%d')))
    # Add markers for broken predicted readings
    fig.add_trace(go.Scatter(x=broken_predicted.index, y=broken_predicted[sensor_name], mode='markers', name='Broken Predicted',
                             marker=dict(color='red', size=12, symbol='x'), hoverinfo='x+y+text',
                             text='Predicted Broken: ' + broken_predicted.index.strftime('%Y-%m-%d')))
    # Update layout for title, legend, and figure size
    fig.update_layout(title=sensor_name, legend=dict(x=0, y=1, traceorder='normal', font=dict(family='sans-serif', size=12, color=CUSTOM_TEXT_COLOR)),
                      width=st.session_state.plot_width, height=st.session_state.plot_height, plot_bgcolor=CUSTOM_BG_COLOR)  # Adjust width and height based on session state
    return fig

def prediction_tab():
    st.title('Predictive Analysis')
    st.sidebar.title('Upload CSV File')
    file = st.sidebar.file_uploader("Upload a CSV file to predict", type=['csv'])
    if file is not None:
        # Read the CSV file
        input_data = pd.read_csv(file)

        # Convert the date column to datetime format and set it as the index
        input_data['Date'] = pd.to_datetime(input_data['Date'])
        input_data.set_index('Date', inplace=True)

        # Read the other dataframe with the actual column
        actual_data = pd.read_csv("FCSV.csv")

        # Add the 'actual column' from the other dataframe to input_data
        input_data['Actual_Machine_Status'] = pd.Series(actual_data['label'].values, index=input_data.index)
        input_data['Actual_Machine_Status'] = ["BROKEN" if pred == 1 else "NORMAL" for pred in input_data['Actual_Machine_Status']]

        # Use the loaded model to make predictions on the input data
        predictions = loaded_model.predict(xgb.DMatrix(input_data.iloc[:, :-1]))

        # Add the predictions as a new column
        input_data['Predicted_Machine_Status'] = ["BROKEN" if pred == 1 else "NORMAL" for pred in predictions]

        # Extract the readings from BROKEN state and resample by daily average
        broken_actual = input_data[input_data['Actual_Machine_Status'] == 'BROKEN']
        broken_predicted = input_data[input_data['Predicted_Machine_Status'] == 'BROKEN']

        st.subheader('Machine Status Report')

        # Apply custom styling to highlight "Normal" in green and "Broken" in red

        # Apply custom styling to highlight "Normal" in green and "Broken" in red
        def apply_color(val):
            # Initialize an empty Series of the same length as val
            color = pd.Series('', index=val.index)
            # Set color to 'red' where val equals 'Broken'
            color[val == 'BROKEN'] = 'red'
            # Set color to 'green' where val equals 'Normal'
            color[val == 'NORMAL'] = 'green'
            return [f'color: {c}' for c in color]

        styled_df = input_data.style.apply(apply_color, subset=['Actual_Machine_Status', 'Predicted_Machine_Status'])

        st.write(styled_df)

        # Dropdown for selecting sensor
        sensor_name = st.selectbox('Select Sensor', input_data.columns[:-2])

        # Slider for adjusting plot size
        st.sidebar.subheader('Plot Size')
        st.sidebar.markdown('Adjust the width and height of the plot')
        st.session_state.plot_width = st.sidebar.slider('Width', min_value=400, max_value=1200, value=800)
        st.session_state.plot_height = st.sidebar.slider('Height', min_value=200, max_value=800, value=400)

        # Generate Plotly figure
        st.subheader('Predicted Machine Status for Sensor Data')
        fig = generate_plotly_html(input_data, broken_actual, broken_predicted, sensor_name)

        # Plotly figure to HTML
        st.plotly_chart(fig)


def main():
    st.set_page_config(page_title='Predictive Maintenance Dashboard', page_icon=':bar_chart:', layout='wide', initial_sidebar_state='expanded')

    tabs = ["Descriptive Analysis","Predictive Analysis", ]
    selected_tab = st.sidebar.radio("Select Tab", tabs, index=0)  # Set index=0 to make Prediction selected by default

    if selected_tab == "Descriptive Analysis":
        descriptive_tab()

    elif     selected_tab == "Predictive Analysis":
        prediction_tab()
if __name__ == '__main__':
    main()
