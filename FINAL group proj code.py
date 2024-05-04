# %%
import dash
from dash import dcc, html, Input, Output, State, dash_table, callback_context
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import yfinance as yf
import pandas as pd
from yahooquery import Ticker
import yahooquery as yq
from datetime import datetime
import socket

def price_indicator(symbol="AAPL"):
    ticker = Ticker(symbol)
    data = ticker.price
    if symbol in data and 'regularMarketOpen' in data[symbol] and 'regularMarketPrice' in data[symbol]:
        open_price = data[symbol]['regularMarketOpen']
        current_price = data[symbol]['regularMarketPrice']
        change = current_price - open_price
        percent_change = (change / open_price) * 100 if open_price != 0 else 0

        delta_fig = go.Figure()
        delta_fig.add_trace(go.Indicator(
            mode="number+delta",
            value=current_price,
            delta={'reference': open_price, 'relative': True, "valueformat": ".2f", 'increasing': {'color': 'green'}, 'decreasing': {'color': 'red'}, 'font': {'size': 75}},
            title={"text": f"{symbol.upper()} Current Price"},
            number={'prefix': "$", 'font': {'size': 150}}  # Increased font size for the number
        ))
        delta_fig.update_layout(
            height=600,  # Adjusted height for better visibility
            width=1200,   # Adjusted width to fit content
            margin={'l': 100, 'r': 100, 't': 150, 'b': 100}  # Adjusted margins to ensure all text is visible
        )
        return delta_fig
    return go.Figure()  # Return an empty figure if data is not available


def candlestick_chart(symbol, period="1y", interval="1d"):
    ticker = Ticker(symbol)
    history = ticker.history(period=period, interval=interval)
    if history.empty:
        return go.Figure()

    # Assuming history is a DataFrame with an index that includes dates
    if 'date' not in history.columns:
        history.reset_index(inplace=True)

    # Get the latest P/E ratio
    pe_ratio = ticker.summary_detail[symbol].get("trailingPE", "N/A")

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=(f'{symbol.upper()} OHLC (P/E: {pe_ratio})', 'Volume'),
                        row_heights=[0.7, 0.3])
    
    fig.add_trace(go.Candlestick(x=history['date'], open=history['open'], high=history['high'],
                                 low=history['low'], close=history['close'], name='OHLC'), row=1, col=1)
    fig.add_trace(go.Bar(x=history['date'], y=history['volume'], name='Volume'), row=2, col=1)
    fig.update_layout(xaxis_rangeslider_visible=False)
    return fig


# Generates html table with the latest articles for company associated with ticker
def generate_news_table(symbol="AAPL", limit=5):
    news_raw = yq.search(symbol)["news"][:limit]
    return html.Table(
        [html.Tr([html.Th("News", colSpan=3)], id="news-table-header")] + 
        [html.Tr(
           [html.Td(html.Img(src=article["thumbnail"]["resolutions"][-1]["url"], height="70px", width="70px") 
                    if "thumbnail" in article else ""),
            html.Td(html.A(article["publisher"] + ": " + article["title"], href=article["link"], 
                           target="_blank", rel="noopener noreferrer")),
            html.Td(datetime.fromtimestamp(article["providerPublishTime"]).strftime("%m/%d/%Y, %H:%M:%S"))
            ]
        ) for article in news_raw],
        className="table"
    )

# Load S&P 500 stock symbols
sp500_df = pd.read_csv('C:/Users/03msk/Downloads/sp-500-index-04-24-2024.csv')
tickers = sp500_df['Symbol'].tolist()

# Initialize data structures
sectors = {}
market_caps = {}
sector_data = {}

# Fetch sector information and market data
for ticker in tickers:
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        sector = info.get('sector', 'Unknown')
        market_cap = info.get('marketCap', 0)
        if sector != 'Unknown':
            if sector not in sectors:
                sectors[sector] = []
                market_caps[sector] = 0
                sector_data[sector] = pd.DataFrame()
            sectors[sector].append((ticker, market_cap, stock))
            market_caps[sector] += market_cap
            hist_data = stock.history(period="1y")
            if not hist_data.empty:
                hist_data.index = hist_data.index.tz_localize(None)
                sector_data[sector] = pd.concat([sector_data[sector], hist_data[['Close']]], axis=1)
    except Exception as e:
        print(f"Failed to fetch data for {ticker}: {e}")

# Calculate returns
sector_returns = {}
start_of_year = datetime(datetime.now().year, 1, 1)
for sector, data in sector_data.items():
    if not data.empty:
        data.fillna(method='ffill', inplace=True)
        daily_return = data.pct_change().iloc[-1].mean()
        ytd_data = data[data.index >= start_of_year]
        ytd_return = (ytd_data.iloc[-1] / ytd_data.iloc[0] - 1).mean()
        sector_returns[sector] = {'Daily Return': daily_return, 'YTD Return': ytd_return}

# Create the Dash app
app = dash.Dash(__name__, suppress_callback_exceptions=True)

# App layout
app.layout = html.Div([
    html.H1("Real-Time Stock Market Dashboard", style={'text-align': 'center'}),
    dcc.Input(id='stock-input', type='text', value='', placeholder="Enter stock ticker", className="input-field"),
    html.Button('Search', id='search-button', n_clicks=0, className="button"),
    dcc.Graph(id='price-metric', style={'display': 'none'}),  # Initially hidden
    dcc.Graph(id='price-history', style={'display': 'none'}),  # Initially hidden
    html.Div(id='news-output', style={'display': 'none'}),  # Initially hidden,
    html.Br(),
    dcc.Graph(id='sector-pie', figure=px.pie(pd.DataFrame(list(market_caps.items()), columns=['Sector', 'Market Cap']),
                                             values='Market Cap', names='Sector', title='Market Weight by Sector')),
    html.Div(id='sector-output'),
    html.Div([
        dash_table.DataTable(
            id='stock-table',
            columns=[{"name": i, "id": i} for i in ["Ticker", "Market Cap"]],
            data=[]
        )
    ], id='table-container', style={'display': 'none'})  # Wrap the DataTable and control visibility here
])


# Callback to update the stock data based on the user's search
@app.callback(
    [Output('price-history', 'figure'),
     Output('price-metric', 'figure'),
     Output('news-output', 'children'),
     Output('price-history', 'style'),  # Control visibility of the price history chart
     Output('price-metric', 'style'),   # Control visibility of the price metric chart
     Output('news-output', 'style')],   # Control visibility of the news output
    [Input('search-button', 'n_clicks')],
    [State('stock-input', 'value')]
)
def update_output_div(n_clicks, symbol):
    if symbol.strip():  # Check if symbol is not just empty or spaces
        # Get the price chart, price indicator, and news
        price_history_fig = candlestick_chart(symbol)
        price_metric_fig = price_indicator(symbol)
        news_content = generate_news_table(symbol)

        # Check if figures are empty (i.e., no data was returned)
        if not price_history_fig.data and not price_metric_fig.data:
            return go.Figure(), go.Figure(), "No data available.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}
        
        return price_history_fig, price_metric_fig, news_content, {'display': 'block'}, {'display': 'block'}, {'display': 'block'}
    return go.Figure(), go.Figure(), "Enter a valid stock symbol and click search.", {'display': 'none'}, {'display': 'none'}, {'display': 'none'}


def update_stock_data(n_clicks, symbol):
    if symbol:
        stock_info = price_indicator(symbol)  # Get the current stock price indicator
        fig = candlestick_chart(symbol, "1y", "1d")  # Get the historical price chart
        news_content = generate_news_table(symbol)  # Get the latest news
        return (
            dcc.Graph(figure=stock_info),
            news_content,
            {'display': 'block'},
            fig
        )
    return ("Enter a valid stock symbol.", html.Div(), {'display': 'none'}, go.Figure())

# Callback for sector interaction
@app.callback(
    [Output('sector-output', 'children'),
     Output('stock-table', 'data'),
     Output('table-container', 'style')],  # Update to target the container Div
    Input('sector-pie', 'clickData')
)
def display_top_stocks(clickData):
    if clickData:
        sector_name = clickData['points'][0]['label']
        top_stocks = sorted(sectors[sector_name], key=lambda x: x[1], reverse=True)[:10]
        daily_return = sector_returns.get(sector_name, {}).get('Daily Return', 0)
        ytd_return = sector_returns.get(sector_name, {}).get('YTD Return', 0)
        table_data = [{"Ticker": stock[0], "Market Cap": f"{stock[1]:,}"} for stock in top_stocks]
        return (
            [html.H3(f"Daily Return for {sector_name}: {daily_return:.2%}"),
             html.H3(f"YTD Return for {sector_name}: {ytd_return:.2%}")],
            table_data,
            {'display': 'block'}  # Now correctly targets the container Div
        )
    return "Click on a sector in the pie chart to see the top 10 stocks.", [], {'display': 'none'}

# Function to find a free port for the server
def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]

if __name__ == '__main__':
    port = find_free_port()
    web = f"http://127.0.0.1:{port}"
    print(web)
    app.run_server(debug=True, port=port)

# %%
