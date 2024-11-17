"""Monte Carlo Race Simulation Analysis and Visualization Module

This module provides a set of functions for analyzing and visualizing the results of Monte Carlo
race simulations. It uses Plotly for creating interactive visualizations, to help interpret simulation
data and gain insights into the estimated effectiveness of different race strategies and estimated driver performance.
"""

import pandas as pd
from plotly import graph_objects as go, express as px
from RaceStrategyEngine.utility import count_driver_simulations


def plot_drivers_finishing_position_frequency(pandas_df: pd.DataFrame):
    """Generate an interactive bar plot showing the frequency of finishing positions for each driver.

    This function creates a plot where each bar represents the number of times a driver finished in a specific position
    across all simulations. The plot includes a dropdown menu to switch between different drivers.

    Args:
        pandas_df (pd.DataFrame): A DataFrame containing simulation results. Must include 'driver' and 'position' columns.
    """
    drivers = pandas_df['driver'].unique()

    fig = go.Figure()

    for driver in drivers:
        driver_df = pandas_df[pandas_df['driver'] == driver]
        positions_frequency = driver_df['position'].value_counts().sort_index()

        driver_fig = px.bar(x=positions_frequency.index,
                            y=positions_frequency.values,
                            labels={'x': 'Position', 'y': 'Frequency'},
                            color=positions_frequency.values,
                            text=positions_frequency.values)

        # Add traces from the px figure to the main figure
        for trace in driver_fig.data:
            trace.visible = (driver == drivers[0])  # Only first driver visible by default
            fig.add_trace(trace)

    # Create and add dropdown
    update_menu = [dict(
        active=0,
        buttons=[dict(label=driver,
                      method='update',
                      args=[{'visible': [d == driver for d in drivers for _ in driver_fig.data]},
                            {
                                'title': f"Frequency of End of Race Result for {driver} "
                                         f"({count_driver_simulations(pandas_df, driver):,} Simulations)"}])
                 for driver in drivers],
        direction="down",
        pad={"r": 10, "t": 10},
        showactive=True,
        x=1.0,
        xanchor="right",
        y=1.1,
        yanchor="top"
    )]

    fig.update_layout(
        updatemenus=update_menu,
        title_text=f"Frequency of End of Race Result for {drivers[0]}"
                   f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
        xaxis_title="Position",
        yaxis_title="Frequency",
        coloraxis_colorbar=dict(title="Frequency"),
        xaxis=dict(
            dtick=1,
            tickmode='linear'
        ),
    )

    fig.update_traces(
        textposition='outside',
        hovertemplate="Position: %{x}<br>Frequency: %{y}"
    )

    fig.show()


def plot_traffic_heatmap(pandas_df: pd.DataFrame):
    """Create an interactive heatmap showing the mean laps behind traffic for different strategies and drivers.

    This function generates a heatmap where the color intensity represents the mean number of laps spent behind traffic
    for each combination of driver and strategy. A dropdown menu allows switching between drivers.

    Args:
        pandas_df (pd.DataFrame): A DataFrame containing simulation results. Must include 'driver', 'compounds_used',
                                  and 'laps_behind_traffic' columns.

    """

    drivers = pandas_df['driver'].unique()
    fig = go.Figure()

    for driver in drivers:
        driver_df = pandas_df[pandas_df['driver'] == driver]
        traffic_data = driver_df.groupby('compounds_used')['laps_behind_traffic'].mean().reset_index()
        traffic_data = traffic_data.sort_values('laps_behind_traffic', ascending=False)

        fig.add_trace(
            go.Heatmap(
                z=[traffic_data['laps_behind_traffic']],
                y=['Mean Laps Behind Traffic'],
                x=traffic_data['compounds_used'],
                colorscale='YlOrRd',
                name=driver,
                visible=(driver == drivers[0])
            )
        )

    dropdown_buttons = [dict(
        method='update',
        label=driver,
        args=[{'visible': [driver == trace.name for trace in fig.data]},
              {
                  'title': f'Estimated Mean Laps Behind Traffic for Different Strategies for {driver} '
                           f'({count_driver_simulations(pandas_df, driver):,} Simulations)'}]
    ) for driver in drivers]

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=1.0,
            y=1.1,
            xanchor='right',
            yanchor='top'
        )],
        title_text=f"Estimated Mean Laps Behind Traffic for Different Strategies for {drivers[0]}"
                   f" ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)",
        xaxis_title='Race Strategy (compounds used)',
        yaxis_title='',
        xaxis_tickangle=-45,
    )

    fig.show()


def plot_traffic_boxplot(pandas_df: pd.DataFrame):
    """
    Generates an interactive box plot showing the distribution of laps behind traffic for different tyre
    strategies and drivers.

    This method creates an interactive box plot that allows toggling between different drivers to visualize the
    spread and central tendency of laps spent behind traffic for each strategy.

    Args:
        pandas_df (pd.DataFrame): A pandas DataFrame containing the simulation results.

    The plot shows:
    - x-axis represents different tyre compound strategies.
    - y-axis represents the number of laps behind traffic.
    - Each box shows the distribution of laps behind traffic for a strategy.
    - Strategies are sorted in descending order of median laps behind traffic.
    - A dropdown menu allows toggling between different drivers.
    """

    drivers = pandas_df['driver'].unique()

    colors = px.colors.qualitative.Plotly

    fig = go.Figure()

    # Creating separate traces for each driver and strategy combination
    # Only solution i could find that work with different colors and a dropdown
    for i, driver in enumerate(drivers):
        driver_df = pandas_df[pandas_df['driver'] == driver]
        order = driver_df.groupby('compounds_used')['laps_behind_traffic'].median().sort_values(
            ascending=False).index

        for j, strategy in enumerate(order):
            filtered_df = driver_df[driver_df['compounds_used'] == strategy]
            fig.add_trace(go.Box(
                y=filtered_df['laps_behind_traffic'],
                name=f'{driver} - {strategy}',
                marker_color=colors[j % len(colors)],
                visible=(i == 0)  # Make only the first driver visible initially
            ))

    # Create the dropdown menu
    dropdown_buttons = [
        {'label': driver,
         'method': 'update',
         'args': [{'visible': [True if trace.name.startswith(driver) else False for trace in fig.data]},
                  {'title': f'Distribution of Estimated Laps Behind Traffic for Different Strategies for {driver}'
                            f' ({count_driver_simulations(pandas_df, driver):,} Simulations)'}]}
        for driver in drivers
    ]

    # Update layout to include the dropdown menu
    fig.update_layout(
        updatemenus=[{
            'buttons': dropdown_buttons,
            'direction': 'down',
            'showactive': True,
            'x': 1.0,
            'y': 1.1,
            'xanchor': 'right',
            'yanchor': 'top'

        }],
        title_text=f'Distribution of Estimated Laps Behind Traffic for Different Strategies for {drivers[0]}'
                   f' ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)',
        yaxis_title='Laps Behind Traffic',
        xaxis_title='Race Strategy (compounds used)',
        xaxis_tickangle=-45,
        showlegend=False
    )

    fig.show()


def plot_race_strategy_position_distribution(pandas_df: pd.DataFrame):
    """Generate a stacked bar plot showing the distribution of finishing positions for different race strategies.

    This function creates an interactive plot where each bar represents a race strategy, and the segments of the bar
    show the frequency of different finishing positions. A dropdown menu allows switching between drivers.

    Args:
        pandas_df (pd.DataFrame): A DataFrame containing simulation results. Must include 'driver', 'compounds_used',
                                  and 'position' columns.
    """

    drivers = pandas_df['driver'].unique()

    fig = go.Figure()

    traces = []
    for driver in drivers:
        driver_df = pandas_df[pandas_df['driver'] == driver]
        count = driver_df.groupby(['compounds_used', 'position'])['position'].value_counts().reset_index(
            name='frequency')

        # Add traces for each points category
        for points_value in count['position'].unique():
            filtered_data = count[count['position'] == points_value]

            traces.append(go.Bar(
                x=filtered_data['compounds_used'],
                y=filtered_data['frequency'],
                name=f'P{points_value} ({driver} Plot)',
                hovertemplate="Strategy (compounds used): %{x}<br>Frequency: %{y}<br>Position: " + str(points_value),
                visible=True if driver == drivers[0] else False
            ))

    # Add traces to the figure
    fig.add_traces(traces)

    # Create dropdown buttons for driver selection
    dropdown_buttons = [dict(method="update",
                             args=[{"visible": [trace.name.endswith(f"({driver} Plot)") for trace in traces]},
                                   {
                                       'title': f"Race Strategy Position Distribution for {driver}"
                                                f" ({count_driver_simulations(pandas_df, driver):,} Simulations)"}
                                   ],
                             label=driver) for driver in drivers]

    # Update layout to include the dropdown menu and format the plot

    fig.update_layout(
        updatemenus=[dict(active=0,
                          buttons=dropdown_buttons,
                          x=1.0,
                          y=1.1,
                          xanchor='right',
                          yanchor='top')],
        barmode='stack',
        title_text=f'Race Strategy Position Distribution for {drivers[0]} '
                   f'({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)',
        title_x=0.4,
        xaxis_title=" Race Strategy (compounds used)",
        yaxis_title="Sum of Frequency",
        legend_title="Position Achieved",
        legend=dict(font=dict(size=22))
    )

    fig.show()


def plot_drivers_position_dominance(pandas_df: pd.DataFrame):
    """
    Generates an enhanced bar plot showing the percentage and count of times each driver achieved a specific finishing position.

    This method creates a visual representation of how often each driver finishes in a particular position
    across all simulations, with the ability to toggle between different positions.

    Args:
        pandas_df (pd.DataFrame): A pandas DataFrame containing the simulation results.
                                Must include 'driver' and 'position' columns.

    The plot shows:
        - X-axis: Represents the drivers.
        - Y-axis: Shows the percentage of times a driver achieved the selected position.
        - Color: Bars are colored on a gradient scale based on percentage (higher percentages are darker).
        - Text: Displays both the percentage and actual count for each driver.
        - Dropdown: Allows selection of different finishing positions to display.
    """
    positions = pandas_df['position'].unique()
    positions.sort()

    drivers = pandas_df['driver'].unique()
    custom_colors = ['#DC0000', '#FFFF00', '#32CD32', 'pink', 'purple']

    fig = go.Figure()

    for position in positions:
        position_df = pandas_df[pandas_df['position'] == position]
        total_count = len(position_df)
        driver_counts = position_df['driver'].value_counts()
        percentages = driver_counts / total_count * 100

        text = [f'{percentages.get(driver, 0):.3f}%<br>({driver_counts.get(driver, 0)})' for driver in drivers]

        fig.add_trace(go.Bar(
            x=drivers,
            y=[percentages.get(driver, 0) for driver in drivers],
            name=f'P{position}',
            text=text,
            textposition='outside',
            marker=dict(
                color=[percentages.get(driver, 0) for driver in drivers],
                colorscale=custom_colors,
                showscale=True,
                cmin=0,
                cmax=100
            ),
            visible=(position == positions[0])
        ))

    dropdown_buttons = [dict(
        method='update',
        label=f'P{position}',
        args=[{'visible': [pos == position for pos in positions]},
              {'title': f'Distribution of Driver Finishing Positions - P{position}'
                        f' ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)'}]
    ) for position in positions]

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=1.0,
            y=1.1,
            xanchor='right',
            yanchor='top'
        )],
        title_text=f'Distribution of Driver Finishing Positions - P{positions[0]}'
                   f' ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)',
        xaxis_title='Drivers',
        yaxis_title='Percentage',
        yaxis=dict(
            tickformat='.1f',
            ticksuffix='%',
            range=[0, 100]
        ),
        bargap=0.2,
        showlegend=False
    )

    fig.show()


def get_vital_strategy_location(driver, pandas_df):
    """ Helper function that calculate the mean for a driver's race strategies.

    This function computes the mean position for each strategy used by a driver, ranks the strategies,
    and identifies the best and worst strategies based on mean position.

    Args:
        driver (str): The name of the driver to analyze.
        pandas_df (pd.DataFrame): A DataFrame containing simulation results for all drivers.

    Returns:
        tuple: A tuple containing three elements:
            - DataFrame: Grouped data for the driver's strategies, including mean position and rank.
            - tuple: (x, y) coordinates of the best strategy.
            - tuple: (x, y) coordinates of the worst strategy.
    """
    driver_df = pandas_df[pandas_df['driver'] == driver]
    grouped_driver_df = driver_df.groupby('compounds_used').agg({'position': 'mean'}).reset_index()
    grouped_driver_df = grouped_driver_df.sort_values('position')
    grouped_driver_df['rank'] = range(1, len(grouped_driver_df) + 1)  # Start ranking from 1
    best_strategy_x = grouped_driver_df.iloc[0]['rank']
    best_strategy_y = grouped_driver_df.iloc[0]['position']
    worst_strategy_x = grouped_driver_df.iloc[-1]['rank']
    worst_strategy_y = grouped_driver_df.iloc[-1]['position']
    return grouped_driver_df, (best_strategy_x, best_strategy_y), (worst_strategy_x, worst_strategy_y)


def plot_race_strategy_ranking(pandas_df):
    """
    Create an interactive scatter plot showing the ranking of race strategies for each driver.

    This function generates a plot where each point represents a unique race strategy, positioned according to its
    rank and mean finishing position. The plot includes annotations for the best and worst strategies and a dropdown
    menu to switch between drivers.

    Args:
        pandas_df (pd.DataFrame): A DataFrame containing simulation results. Must include 'driver', 'compounds_used',
                                  and 'position' columns.
    """
    drivers = pandas_df['driver'].unique()

    fig = go.Figure()
    custom_colors = ['purple', '#32CD32', 'orange']
    arrow_colour = 'black' 

    position_min = pandas_df['position'].min()
    position_max = pandas_df['position'].max()

    for driver in drivers:
        grouped_df, *_ = get_vital_strategy_location(driver, pandas_df)

        fig.add_trace(
            go.Scatter(
                x=grouped_df['rank'],
                y=grouped_df['position'],
                mode='markers',
                name=driver,
                marker=dict(size=8,
                            color=grouped_df['position'],
                            colorscale=custom_colors,
                            showscale=True,
                            colorbar=dict(title="Mean Position"),
                            symbol='circle',
                            cmin=position_min,
                            cmax=position_max,

                            ),
                text=[f"Strategy: {format_strategy(comp)}<br>Mean Position: {pos:.4f}<br>Est. Rank: {rank}"
                      for comp, pos, rank in zip(grouped_df['compounds_used'], grouped_df['position'], grouped_df['rank'])],
                hoverinfo='text',
                visible=(driver == drivers[0])
            )
        )


        _, best_strategy, worst_strategy = get_vital_strategy_location(drivers[0], pandas_df)
        best_x, best_y = best_strategy
        worst_x, worst_y = worst_strategy


        annotations = [
            dict(x=best_x, y=best_y,
                 text="Est. Best", showarrow=True, arrowhead=2, ax=20, ay=-40, arrowcolor=arrow_colour),
            dict(x=worst_x, y=worst_y,
                 text="Est. Worst", showarrow=True, arrowhead=2, ax=-20, ay=40, arrowcolor=arrow_colour)
        ]

        fig.update_layout(annotations=annotations)


    dropdown_buttons = [dict(
        method='update',
        label=driver,
        args=[{'visible': [d == driver for d in drivers]},
              {'title': f'Average Position for Each Unique Race Strategy for {driver} '
                        f'- Total Unique Strategies: '
                        f'{count_unique_strategies(pandas_df, driver)} '
                        f'({count_driver_simulations(pandas_df, driver):,} Simulations)',
               'annotations': [
                   dict(x=best_strategy[0], y=best_strategy[1],
                        text="Est. Best", showarrow=True, arrowhead=2, ax=20, ay=-40, arrowcolor=arrow_colour),
                   dict(x=worst_strategy[0], y=worst_strategy[1],
                        text="Est. Worst", showarrow=True, arrowhead=2, ax=-20, ay=40, arrowcolor=arrow_colour)
               ]}]
    ) for driver in drivers for _, best_strategy, worst_strategy in
        [get_vital_strategy_location(driver, pandas_df)]]

    fig.update_layout(
        updatemenus=[dict(
            active=0,
            buttons=dropdown_buttons,
            x=1.0,
            y=1.1,
            xanchor='right',
            yanchor='top'
        )],
        title_text=f'Average Position for Each Unique Race Strategy for {drivers[0]}'
                   f' - Total Unique Strategies: '
                   f'{count_unique_strategies(pandas_df, drivers[0])}'
                   f' ({count_driver_simulations(pandas_df, drivers[0]):,} Simulations)',
        xaxis_title='Strategy Rank',
        yaxis_title='Mean Position',
        xaxis=dict(tickmode='auto', tick0=1),  
        yaxis=dict(autorange="reversed")  

    )

    fig.show()


def format_strategy(strategy_str):
    """
    Format a strategy string for better readability.

    This function takes a string representation of a strategy (list of compounds) and formats it
    into a more readable string with compounds separated by ' > '.

    Args:
        strategy_str (str): A string representation of a strategy.

    Returns:
        str: A formatted string representing the strategy.
    """

    compounds = strategy_str.strip("[]'").replace("'", "").split(", ")
    return " > ".join(compounds).strip('()')


def count_unique_strategies(pandas_df: pd.DataFrame, driver:str):
    """ Helper function that count the number of unique strategies used by a specific driver.

    This function groups the data by strategy for a given driver and counts the number of unique strategies.

    Args:
        pandas_df (pd.DataFrame): A DataFrame containing simulation results for all drivers.
        driver (str): The name of the driver to analyze.

    Returns:
        int: The number of unique strategies used by the driver.
    """
    driver_df = pandas_df[pandas_df['driver'] == driver]
    grouped_driver_df = driver_df.groupby('compounds_used').agg({'position': 'mean'}).reset_index()
    count = len(grouped_driver_df)

    return count
