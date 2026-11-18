# YouTube Channel Analytics

A Streamlit dashboard that analyses any public YouTube channel: growth and engagement trends, what upload days and formats perform best, and which videos broke out far beyond the channel's normal.

## Features
- Channel lookup by URL, `@handle`, or channel ID
- Engagement rate per video, Shorts vs long-form comparison, performance by upload day
- **Breakout detection**: flags videos that beat the channel's *median* views by a configurable multiple (median, not mean, so one viral hit doesn't hide the others)

## Tech
Python · Streamlit · YouTube Data API v3 · pandas · Plotly

## Running locally
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # then add your API key(s)
streamlit run app.py
```

## Notes on API quota
The YouTube Data API gives 10,000 units/day. This app avoids `search.list` (100 units per call) and reads each channel's uploads playlist instead (1 unit per call). Results are cached for an hour, so reloading a channel costs nothing.
