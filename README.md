# Global Semiconductor Trade Flows

How did chip and chipmaking-equipment trade change after the US imposed export controls on advanced
computing and semiconductors in October 2022?

Live app: [semiconductors.streamlit.app](https://semiconductors.streamlit.app)

## Overview

An interactive Streamlit dashboard built on UN Comtrade bilateral trade data, 2018 to 2025. It
tracks two product codes:

- **HS 8542, integrated circuits:** exports from China, Japan, South Korea, Taiwan and the USA.
- **HS 8486, semiconductor manufacturing equipment:** exports from Germany, Japan, the Netherlands,
  South Korea and the USA.

It also overlays IC re-exports from four intermediate hubs (Hong Kong, Malaysia, Singapore and
Vietnam). That shows the second hop of the supply chain, where chips pass through a hub on their
way to final assembly markets.

## What the app shows

- **Top 15 exporters** for each product code, ranked across all reporting countries rather than a
  pre-picked list, as a check on which countries matter.
- **Flow map** of exporter-to-importer trade, on a 3D globe or a flat map. Ribbon width scales with
  trade value and uses the same scale across all years, so widths stay comparable when you move the
  year slider.
- **Sankey diagram** of the same flows. The map always includes every destination the Sankey shows,
  so the two views never disagree.
- **Headline metrics:** total tracked IC exports with year-on-year change, and Taiwan's and China's
  shares of the tracked total.
- **Export trends by country** from 2018, with October 2022 marked.

A sidebar sets the year and toggles each exporter and hub on or off.

## Data

UN Comtrade merchandise trade statistics, annual, reporter-side gross exports
([comtradeplus.un.org](https://comtradeplus.un.org)), pulled live through the `comtradeapicall`
API wrapper and cached per session.

## Caveats

- **2025 is incomplete.** China and Taiwan had not yet reported 2025 data when this was built, so
  the analysis is only consistent up to 2024.
- **Gross exports double-count.** A chip that crosses several borders is counted at each one, which
  is why the headline metric is labelled "tracked exports" rather than world exports.
- **Taiwan appears in Comtrade as "Other Asia, nes".** The app relabels it as Taiwan.
- **Shares are of the tracked reporters only.** Taiwan's share of the five tracked IC exporters is
  higher than its share of world IC exports.
- **A few partners cannot be mapped.** Partners with no coordinates on file are left off the map.
  The app lists them when their flows exceed USD 1 billion.

## Tech stack

- **Data:** UN Comtrade API via `comtradeapicall`
- **Processing:** `pandas`, `numpy`
- **Visualisation:** `plotly` (flow map, Sankey, charts)
- **Deployment:** Streamlit Cloud

## Running locally

1. Clone the repository.
2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Add your UN Comtrade subscription key to `.streamlit/secrets.toml`:
   ```toml
   COMTRADE_KEY = "your-key-here"
   ```
4. Run the app:
   ```
   streamlit run app_semi.py
   ```

The deployed app reads `COMTRADE_KEY` from Streamlit Cloud secrets. The key is never stored in the
repository.

## Author

Ng Ka Wai
