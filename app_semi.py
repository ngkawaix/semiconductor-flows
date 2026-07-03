import streamlit as st
import comtradeapicall as comtrade
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import math
import numpy as np

st.set_page_config(
    page_title="Semiconductor Supply Chain Intelligence",
    layout="wide",
    page_icon="🔬"
)

# ── Data loading ───────────────────────────────────────────────────────────
# Canonical Comtrade → display name mapping applied uniformly across all loaders
_COMTRADE_RENAME = {
    'Other Asia, nes':      'Taiwan',        # Comtrade lists Taiwan under this code
    'China, Hong Kong SAR': 'Hong Kong',     # Shorten for map labels
    'Viet Nam':             'Vietnam',       # Localise spelling
}
_COMTRADE_COLS = ['reporterDesc','reporterISO','partnerDesc','partnerISO','period','primaryValue']
YEARS_STR = ','.join(str(y) for y in range(2018, 2026))   # 2018-2025, one API call (8 periods, cap is 12)

# Reporter code lists — named once, reused by every loader that needs them
IC_REPORTER_CODES    = '156,490,410,842,392'   # China, Taiwan, S.Korea, USA, Japan
EQUIP_REPORTER_CODES = '528,842,392,276,410'   # Netherlands, USA, Japan, Germany, S.Korea
HUB_REPORTER_CODES   = '344,458,702,704'       # Hong Kong, Malaysia, Singapore, Vietnam

def _fix_taiwan_iso(df):
    """'Other Asia, nes' (Comtrade code 490 = Taiwan) has no standard ISO3 —
    reporterISO/partnerISO come back blank/non-standard for these rows, so
    after _COMTRADE_RENAME maps the *Desc to 'Taiwan', the row still fails
    the inner-join to coords_df (keyed on real ISO3 'TWN'). Fix the ISO
    columns directly off the already-renamed Desc columns so the merge
    succeeds regardless of whatever raw ISO Comtrade returned."""
    df.loc[df['reporterDesc'] == 'Taiwan', 'reporterISO'] = 'TWN'
    df.loc[df['partnerDesc']  == 'Taiwan', 'partnerISO']  = 'TWN'
    return df

@st.cache_data
def fetch_comtrade(cmd_code, reporter_code=None, partner_code=None, years=YEARS_STR):
    """Generic, cached Comtrade pull. Streamlit caches on the exact argument
    combination, so calling this with different params for different views
    never re-hits the API for a combination already fetched this session.
      reporter_code=None      → all reporters (needed for exporter ranking)
      partner_code=None       → full bilateral detail (needed for map/Sankey)
      partner_code='0'        → World aggregate only (needed for ranking —
                                 this is what makes the pull a ranking pull)
    """
    key = st.secrets["COMTRADE_KEY"]
    return comtrade.getFinalData(
        key, typeCode='C', freqCode='A', clCode='HS', period=years,
        reporterCode=reporter_code, cmdCode=cmd_code, flowCode='X',
        partnerCode=partner_code, partner2Code=None, customsCode=None, motCode=None,
        maxRecords=250000, format_output='JSON', aggregateBy=None,
        breakdownMode='classic', countOnly=None, includeDesc=True
    )

@st.cache_data
def clean_comtrade(df, keep_world_rows=False):
    """Shared post-processing for every Comtrade pull: column selection,
    renaming, Taiwan ISO fix, numeric coercion.
      keep_world_rows=False (default) drops partner='World' rows (ISO 'W00')
      — needed for bilateral views (map/Sankey), where every row must have
      a real destination to draw an arc to.
      keep_world_rows=True keeps them — needed for the exporter ranking,
      where the World-aggregate row IS the number we want (each reporter's
      total exports to World)."""
    df = df[_COMTRADE_COLS].copy()
    if not keep_world_rows:
        df = df[df['partnerISO'] != 'W00']
    df['reporterDesc'] = df['reporterDesc'].replace(_COMTRADE_RENAME)
    df['partnerDesc']  = df['partnerDesc'].replace(_COMTRADE_RENAME)
    df = _fix_taiwan_iso(df)
    df['primaryValue'] = pd.to_numeric(df['primaryValue'], errors='coerce')
    return df

def load_global_data():
    return clean_comtrade(fetch_comtrade('8542', reporter_code=IC_REPORTER_CODES))

def load_global_equip_data():
    """HS 8486: semiconductor equipment exports from the five major tool makers."""
    return clean_comtrade(fetch_comtrade('8486', reporter_code=EQUIP_REPORTER_CODES))

def load_reexport_data():
    """HS 8542: IC exports from key intermediate re-export / transit hubs.
    These countries receive chips from primary exporters and onward-ship
    to final assembly markets. Showing their outbound flows reveals the
    second hop of the supply chain (Singapore → Mexico, HK → SE Asia, etc.)
    """
    return clean_comtrade(fetch_comtrade('8542', reporter_code=HUB_REPORTER_CODES))

def load_exporter_ranking(cmd_code):
    """ALL reporters, World-aggregate only — each country's total exports to
    the world for this HS code. This is the input for the top-N ranking bar
    chart, and it's deliberately NOT restricted to the pre-picked country
    lists above: the point is to check who actually shows up, not re-display
    who we already chose."""
    return clean_comtrade(
        fetch_comtrade(cmd_code, reporter_code=None, partner_code='0'),
        keep_world_rows=True,
    )

@st.cache_data
def rank_top_exporters(df, n=15, max_year=2024):
    """Cumulative ranking across years (not year-by-year): sum each
    reporter's exports over the window, then express every reporter's total
    as a percentage share of the sum of ALL reporters' totals. E.g. if
    Taiwan's summed exports are 500 and every reporter's summed exports
    together add up to 2000, Taiwan's share_pct is 500 / 2000 * 100 = 25%.
    Excludes 2025 by default — reporting coverage for these HS codes
    collapses in 2025 (see earlier coverage-check work in the notebook)."""
    d = df[df['period'].astype(int) <= max_year]
    ranked = (
        d.groupby('reporterDesc')['primaryValue'].sum()
        .sort_values(ascending=False)
        .reset_index()
    )
    ranked['share_pct'] = 100 * ranked['primaryValue'] / ranked['primaryValue'].sum()
    ranked['value_usd_bn'] = ranked['primaryValue'] / 1e9
    return ranked.head(n)

# ── Canonical colour palette ───────────────────────────────────────────────
# Single canonical palette used by BOTH maps, both projections, and every
# Sankey/time-series in Tab 1. Saturation chosen for legibility on the white
# ocean: pale tints (old Japan/Netherlands/Germany) are darkened.
country_hex = {
    'Taiwan':        '#1D91C0',   # blue
    'Rep. of Korea': '#0D9488',   # teal-600
    'China':         '#DC2626',   # red — explicit, for instant recognition
    'Netherlands':   '#EA580C',   # orange-600
    'USA':           '#225EA8',   # deep blue
    'Japan':         '#65A30D',   # lime-600
    'Germany':       '#7C3AED',   # violet-600
}
country_rgba = {
    'Taiwan':        [29,  145, 192, 210],
    'Rep. of Korea': [13,  148, 136, 210],
    'China':         [220,  38,  38, 210],
    'Netherlands':   [234,  88,  12, 210],
    'USA':           [34,   94, 168, 210],
    'Japan':         [101, 163,  13, 210],
    'Germany':       [124,  58, 237, 210],
}
IC_EXPORTERS    = ['China', 'Japan', 'Rep. of Korea', 'Taiwan', 'USA']
EQUIP_EXPORTERS = ['Germany', 'Japan', 'Netherlands', 'Rep. of Korea', 'USA']
src_hex       = {k: country_hex[k] for k in IC_EXPORTERS}
equip_src_hex = {k: country_hex[k] for k in EQUIP_EXPORTERS}

# ── Re-export hub palette ───────────────────────────────────────────────────
# Intermediate transit / OSAT hubs that receive chips from primary exporters
# and forward them to final assembly markets.  Distinct from the main palette.
REEXPORT_HUBS = ['Hong Kong', 'Malaysia', 'Singapore', 'Vietnam']   # alphabetical
hub_hex = {
    'Hong Kong': '#EC4899',   # pink-500
    'Malaysia':  '#10B981',   # emerald-500
    'Singapore': '#F59E0B',   # amber-500
    'Vietnam':   '#8B5CF6',   # purple-500
}
hub_rgba = {
    'Hong Kong': [236,  72, 153, 210],
    'Malaysia':  [ 16, 185, 129, 210],
    'Singapore': [245, 158,  11, 210],
    'Vietnam':   [139,  92, 246, 210],
}
# Unified lookup used by Sankey destination coloring so the SAME country
# always gets the same colour regardless of which Sankey it appears in.
ALL_COUNTRY_HEX = {**country_hex, **hub_hex}
# ── Helpers ────────────────────────────────────────────────────────────────
country_coords = {
    'ABW': (12.5211, -69.9683),  'AFG': (33.9391, 67.7100),   'AGO': (-11.2027, 17.8739),
    'AIA': (18.2206, -63.0686),  'ALB': (41.1533, 20.1683),   'AND': (42.5462, 1.6016),
    'ARE': (23.4241, 53.8478),   'ARG': (-38.4161, -63.6167), 'ARM': (40.0691, 45.0382),
    'ASM': (-14.2710, -170.1322), 'ATA': (-75.2509, -0.0713),  'ATF': (-49.2803, 69.3485),
    'ATG': (17.0608, -61.7964),  'AUS': (-25.2740, 133.7751), 'AUT': (47.5162, 14.5501),
    'AZE': (40.1431, 47.5769),   'BDI': (-3.3731, 29.9189),   'BEL': (50.5039, 4.4699),
    'BEN': (9.3077, 2.3158),     'BES': (12.1784, -68.2385),  'BFA': (12.2383, -1.5616),
    'BGD': (23.6850, 90.3563),   'BGR': (42.7339, 25.4858),   'BHR': (26.0667, 50.5577),
    'BHS': (25.0343, -77.3963),  'BIH': (43.9159, 17.6791),   'BLM': (17.9000, -62.8333),
    'BLR': (53.7098, 27.9534),   'BLZ': (17.1899, -88.4976),  'BMU': (32.3214, -64.7574),
    'BOL': (-16.2902, -63.5887), 'BRA': (-14.2350, -51.9253), 'BRB': (13.1939, -59.5432),
    'BRN': (4.5353, 114.7277),   'BTN': (27.5142, 90.4336),   'BVT': (-54.4232, 3.4132),
    'BWA': (-22.3285, 24.6849),  'CAF': (6.6111, 20.9394),    'CAN': (56.1304, -106.3468),
    'CCK': (-12.1642, 96.8710),  'CHE': (46.8182, 8.2275),    'CHL': (-35.6751, -71.5430),
    'CHN': (35.8617, 104.1954),  'CIV': (7.5400, -5.5471),    'CMR': (7.3697, 12.3547),
    'COD': (-4.0383, 21.7587),   'COG': (-0.2280, 15.8277),   'COK': (-21.2367, -159.7777),
    'COL': (4.5709, -74.2973),   'COM': (-11.8750, 43.8722),  'CPV': (16.0022, -24.0132),
    'CRI': (9.7489, -83.7534),   'CUB': (21.5218, -77.7812),  'CUW': (12.1696, -68.9900),
    'CXR': (-10.4475, 105.6904), 'CYM': (19.5135, -80.5669),  'CYP': (35.1264, 33.4299),
    'CZE': (49.8175, 15.4730),   'DEU': (51.1657, 10.4515),   'DJI': (11.8251, 42.5903),
    'DMA': (15.4150, -61.3710),  'DNK': (56.2639, 9.5018),    'DOM': (18.7357, -70.1627),
    'DZA': (28.0339, 1.6596),    'ECU': (-1.8312, -78.1834),  'EGY': (26.8206, 30.8025),
    'ERI': (15.1794, 39.7823),   'ESH': (24.2155, -12.8858),  'ESP': (40.4637, -3.7492),
    'EST': (58.5953, 25.0136),   'ETH': (9.1450, 40.4897),    'FIN': (61.9241, 25.7482),
    'FJI': (-17.7134, 178.0650), 'FLK': (-51.7963, -59.5236), 'FRA': (46.2276, 2.2137),
    'FRO': (62.0079, -6.7858),   'FSM': (7.4256, 150.5508),   'GAB': (-0.8037, 11.6094),
    'GBR': (55.3781, -3.4360),   'GEO': (42.3154, 43.3569),   'GGY': (49.4657, -2.5853),
    'GHA': (7.9465, -1.0232),    'GIB': (36.1408, -5.3536),   'GIN': (9.9456, -9.6966),
    'GLP': (16.2650, -61.5510),  'GMB': (13.4432, -15.3101),  'GNB': (11.8037, -15.1804),
    'GNQ': (1.6508, 10.2679),    'GRC': (39.0742, 21.8243),   'GRD': (12.2628, -61.6042),
    'GRL': (71.7069, -42.6043),  'GTM': (15.7835, -90.2308),  'GUF': (3.9339, -53.1258),
    'GUM': (13.4443, 144.7937),  'GUY': (4.8604, -58.9302),   'HKG': (22.3193, 114.1694),
    'HMD': (-53.0818, 73.5042),  'HND': (15.1999, -86.2419),  'HRV': (45.1000, 15.2000),
    'HTI': (18.9712, -72.2852),  'HUN': (47.1625, 19.5033),   'IDN': (-0.7893, 113.9213),
    'IMN': (54.2361, -4.5481),   'IND': (20.5937, 78.9629),   'IOT': (-6.3432, 71.8765),
    'IRL': (53.1424, -7.6921),   'IRN': (32.4279, 53.6880),   'IRQ': (33.2232, 43.6793),
    'ISL': (64.9631, -19.0208),  'ISR': (31.0461, 34.8516),   'ITA': (41.8719, 12.5674),
    'JAM': (18.1096, -77.2975),  'JEY': (49.2144, -2.1312),   'JOR': (30.5852, 36.2384),
    'JPN': (36.2048, 138.2529),  'KAZ': (48.0196, 66.9237),   'KEN': (-0.0236, 37.9062),
    'KGZ': (41.2044, 74.7661),   'KHM': (12.5657, 104.9910),  'KIR': (-3.3704, -168.7340),
    'KNA': (17.3578, -62.7830),  'KOR': (35.9078, 127.7669),  'KWT': (29.3117, 47.4818),
    'LAO': (19.8563, 102.4955),  'LBN': (33.8547, 35.8623),   'LBR': (6.4281, -9.4295),
    'LBY': (26.3351, 17.2283),   'LCA': (13.9094, -60.9789),  'LIE': (47.1660, 9.5554),
    'LKA': (7.8731, 80.7718),    'LSO': (-29.6099, 28.2336),  'LTU': (55.1694, 23.8813),
    'LUX': (49.8153, 6.1296),    'LVA': (56.8796, 24.6032),   'MAC': (22.1987, 113.5439),
    'MAF': (18.0708, -63.0501),  'MAR': (31.7917, -7.0926),   'MCO': (43.7384, 7.4246),
    'MDA': (47.4116, 28.3699),   'MDG': (-18.7669, 46.8691),  'MDV': (3.2028, 73.2207),
    'MEX': (23.6345, -102.5528), 'MHL': (7.1315, 171.1845),   'MKD': (41.6086, 21.7453),
    'MLI': (17.5707, -3.9962),   'MLT': (35.9375, 14.3754),   'MMR': (21.9162, 95.9560),
    'MNE': (42.7087, 19.3744),   'MNG': (46.8625, 103.8467),  'MNP': (15.0979, 145.6739),
    'MOZ': (-18.6657, 35.5296),  'MRT': (21.0079, -10.9408),  'MSR': (16.7425, -62.1874),
    'MTQ': (14.6415, -61.0242),  'MUS': (-20.3484, 57.5522),  'MWI': (-13.2543, 34.3015),
    'MYS': (4.2105, 101.9758),   'MYT': (-12.8275, 45.1662),  'NAM': (-22.9575, 18.4904),
    'NCL': (-20.9043, 165.6180), 'NER': (17.6078, 8.0817),    'NFK': (-29.0408, 167.9547),
    'NGA': (9.0820, 8.6753),     'NIC': (12.8654, -85.2072),  'NIU': (-19.0544, -169.8672),
    'NLD': (52.1326, 5.2913),    'NOR': (60.4720, 8.4689),    'NPL': (28.3949, 84.1240),
    'NRU': (-0.5228, 166.9315),  'NZL': (-40.9006, 174.8860), 'OMN': (21.5126, 55.9233),
    'PAK': (30.3753, 69.3451),   'PAN': (8.5380, -80.7821),   'PCN': (-24.7036, -127.4393),
    'PER': (-9.1900, -75.0152),   'PHL': (12.8797, 121.7740),  'PLW': (7.5150, 134.5825),
    'PNG': (-6.3150, 143.9555),  'POL': (51.9194, 19.1451),   'PRI': (18.2208, -66.5901),
    'PRK': (40.3399, 127.5101),  'PRT': (39.3999, -8.2245),   'PRY': (-23.4425, -58.4438),
    'PSE': (31.9522, 35.2332),   'PYF': (-17.6797, -149.4068), 'QAT': (25.3548, 51.1839),
    'REU': (-21.1151, 55.5364),  'ROU': (45.9432, 24.9668),   'RUS': (61.5240, 105.3188),
    'RWA': (-1.9403, 29.8739),   'SAU': (23.8859, 45.0792),   'SDN': (12.8628, 30.2176),
    'SEN': (14.4974, -14.4524),  'SGP': (1.3521, 103.8198),   'SGS': (-54.4296, -36.5879),
    'SHN': (-24.1435, -10.0307), 'SJM': (77.5536, 23.6703),   'SLB': (-9.6457, 160.1562),
    'SLE': (8.4606, -11.7799),   'SLV': (13.7942, -88.8965),  'SMR': (43.9424, 12.4578),
    'SOM': (5.1521, 46.1996),    'SPM': (46.8852, -56.3159),  'SRB': (44.0165, 21.0059),
    'SSD': (6.8770, 31.3070),    'STP': (0.1864, 6.6131),     'SUR': (3.9193, -56.0278),
    'SVK': (48.6690, 19.6990),   'SVN': (46.1512, 14.9955),   'SWE': (60.1282, 18.6435),
    'SWZ': (-26.5225, 31.4659),  'SXM': (18.0425, -63.0548),  'SYC': (-4.6796, 55.4920),
    'SYR': (34.8021, 38.9968),   'TCA': (21.6940, -71.7979),  'TCD': (15.4542, 18.7322),
    'TGO': (8.6195, 0.8248),     'THA': (15.8700, 100.9925),  'TJK': (38.8610, 71.2761),
    'TKL': (-9.2002, -171.8484), 'TKM': (38.9697, 59.5563),   'TLS': (-8.8742, 125.7275),
    'TON': (-21.1789, -175.1982), 'TTO': (10.6918, -61.2225),  'TUN': (33.8869, 9.5375),
    'TUR': (38.9637, 35.2433),   'TUV': (-7.1095, 177.6493),  'TWN': (23.6978, 120.9605),
    'TZA': (-6.3690, 34.8888),   'UGA': (1.3733, 32.2903),    'UKR': (48.3794, 31.1656),
    'UMI': (19.2833, 166.6167),  'URY': (-32.5228, -55.7658), 'USA': (37.0902, -95.7129),
    'UZB': (41.3775, 64.5853),   'VAT': (41.9029, 12.4534),   'VCT': (12.9843, -61.2872),
    'VEN': (6.4238, -66.5897),   'VGB': (18.4207, -64.6399),  'VIR': (18.3358, -64.8963),
    'VNM': (14.0583, 108.2772),  'VUT': (-15.3767, 166.9592), 'WLF': (-13.7687, -177.1560),
    'WSM': (-13.7590, -172.1046), 'YEM': (15.5527, 48.5164),   'ZAF': (-30.5595, 22.9375),
    'ZMB': (-13.1339, 27.8493),   'ZWE': (-19.0154, 29.1549)
}

# Built once at import time (country_coords is static) — avoids rebuilding
# this small but frequently-merged-on DataFrame from a 250-entry dict on
# every Streamlit rerun (the whole script reruns on any widget interaction,
# even ones in other tabs).
coords_df = (
    pd.DataFrame.from_dict(country_coords, orient='index', columns=['lat','lon'])
    .reset_index().rename(columns={'index':'ISO'})
)

def fmt(val):
    if pd.isna(val):  return 'N/A'
    if val >= 1e12:   return f"${val/1e12:.2f}T"
    elif val >= 1e9:  return f"${val/1e9:.1f}B"
    else:             return f"${val/1e6:.0f}M"

def hex_to_rgba(hex_color, alpha=0.45):
    h = hex_color.lstrip('#')
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f'rgba({r},{g},{b},{alpha})'

@st.cache_data
def build_arc_df(df_in):
    """Merge a Comtrade flow table onto coords_df (source + target lat/lon)
    and collapse to one row per (reporter, partner, period). This spans
    ALL years/reporters in df_in and previously ran on every Streamlit
    rerun regardless of which year/tab the user was on — caching it means
    it only recomputes when the underlying loaded data actually changes."""
    df_out = (
        df_in
        .merge(coords_df.rename(columns={'ISO':'reporterISO','lat':'source_lat','lon':'source_lon'}), on='reporterISO', how='inner')
        .merge(coords_df.rename(columns={'ISO':'partnerISO', 'lat':'target_lat', 'lon':'target_lon'}), on='partnerISO',  how='inner')
    )
    return (
        df_out
        .groupby(['reporterDesc','reporterISO','partnerDesc','partnerISO',
                  'period','source_lat','source_lon','target_lat','target_lon'])
        ['primaryValue'].sum().reset_index()
    )

# ── Flow-map rendering (Plotly geo) ─────────────────────────────────────────
# Implementation note: earlier versions used pydeck. The flat MapView broke
# flows at the antimeridian, and deck.gl's experimental _GlobeView could not
# reliably occlude flows on the far side of the planet (they bled through as
# phantom "latitude lines"). Plotly's geo projections solve both by
# construction: the orthographic globe CLIPS anything beyond the horizon,
# and it ships with built-in land/ocean styling — no external GeoJSON, no
# iframe, no basemap dependency. Both projections share one styling dict so
# the 2-D and 3-D views are always identical.

OCEAN_HEX  = "#FFFFFF"
LAND_HEX   = "#B9C4D1"
BORDER_HEX = "#FFFFFF"
FRAME_HEX  = "#E2E8F0"


def _wrap_lon(lon):
    return ((lon + 180.0) % 360.0) - 180.0


def _flow_path(lat1, lon1, lat2, lon2, bow_deg, n=60):
    """
    Quadratic Bezier from (lat1,lon1) to (lat2,lon2), bowed sideways by
    bow_deg degrees, returned as a list of (wrapped_lon, lat) tuples.

    Bezier formula (t goes 0 → 1 along the curve):
        P(t) = (1-t)^2 * P0  +  2(1-t)t * C  +  t^2 * P1
    where P0 = source, P1 = target, and C = control point = the midpoint
    pushed perpendicular to the straight line by bow_deg. Positive bow bows
    the curve to the LEFT of the direction of travel, so all flows share a
    consistent, organised sweep. The target longitude is first "unwrapped"
    to its nearest representation so trans-Pacific flows take the short way.
    """
    if lon2 - lon1 > 180:
        lon2 -= 360
    elif lon2 - lon1 < -180:
        lon2 += 360

    dx, dy = lon2 - lon1, lat2 - lat1
    dist = math.hypot(dx, dy)
    if dist < 1e-6:
        return []

    px, py = -dy / dist, dx / dist
    cx = (lon1 + lon2) / 2 + px * bow_deg
    cy = (lat1 + lat2) / 2 + py * bow_deg

    pts = []
    for i in range(n + 1):
        t = i / n
        lon = (1 - t) ** 2 * lon1 + 2 * (1 - t) * t * cx + t ** 2 * lon2
        lat = (1 - t) ** 2 * lat1 + 2 * (1 - t) * t * cy + t ** 2 * lat2
        pts.append((_wrap_lon(lon), max(-85.0, min(85.0, lat))))
    return pts


@st.cache_data
def build_flow_fig(df, globe=True, height=620,
                   color_col='color', width_col='width'):
    """
    Build the full flow map as a Plotly geo figure:
      per flow — a soft halo line, a saturated core line, and an arrowhead
      marker at the destination (oriented along the path via angleref);
      plus exporter dots/labels and importer dots/labels.

    Flows converging on the same destination get staggered bow magnitudes
    (1.0×, 1.35×, 0.65×, 1.7× …) so the ribbons fan apart mid-flight
    instead of stacking into one rope.
    """
    fig = go.Figure()

    # Per-destination rank → bow multiplier (fan-out of converging flows)
    bow_mult = {}
    for _, grp in df.groupby('partnerISO'):
        grp_sorted = grp.sort_values(width_col, ascending=False)
        for rank, idx in enumerate(grp_sorted.index):
            step = (rank + 1) // 2 * 0.35
            bow_mult[idx] = 1.0 + step if rank % 2 == 1 else 1.0 - step

    flows = []
    for idx, row in df.iterrows():
        dlon = row['target_lon'] - row['source_lon']
        if dlon > 180:    dlon -= 360
        elif dlon < -180: dlon += 360
        dist = math.hypot(dlon, row['target_lat'] - row['source_lat'])
        bow  = max(1.2, min(dist * 0.12, 11.0)) * bow_mult.get(idx, 1.0)

        # Adaptive curve resolution: short hops (e.g. Korea→Japan, a few
        # degrees) render identically with far fewer points than long
        # transoceanic flows, but every point becomes an SVG vertex on
        # BOTH the halo and core traces — fewer points = lighter figure
        # and snappier globe rotation, with no visible change in shape.
        n = int(np.clip(round(dist * 0.6), 16, 60))

        pts = _flow_path(row['source_lat'], row['source_lon'],
                         row['target_lat'], row['target_lon'], bow_deg=bow, n=n)
        if not pts:
            continue

        # Insert a None break where the wrapped path jumps the antimeridian
        lons, lats = [], []
        for j, (lon, lat) in enumerate(pts):
            if j and lons[-1] is not None and abs(lon - lons[-1]) > 180:
                lons.append(None); lats.append(None)
            lons.append(lon); lats.append(lat)

        r, g, b = list(row[color_col])[:3]
        flows.append(dict(
            lons=lons, lats=lats, rgb=(r, g, b),
            width=float(row[width_col]),
            hover=f"{row.get('reporterDesc','')} → {row.get('partnerDesc','')}"
                  f"<br><b>{row.get('value_fmt','')}</b>",
        ))

    for f in flows:                                            # halo pass
        r, g, b = f['rgb']
        fig.add_trace(go.Scattergeo(
            lon=f['lons'], lat=f['lats'], mode='lines',
            line=dict(width=f['width'] * 2.2, color=f"rgba({r},{g},{b},0.22)"),
            hoverinfo='skip', showlegend=False,
        ))
    for f in flows:                                            # core + arrow
        r, g, b = f['rgb']
        n = len(f['lons'])
        sizes = [0] * n
        sizes[-1] = max(9, f['width'] * 2.4 + 5)
        fig.add_trace(go.Scattergeo(
            lon=f['lons'], lat=f['lats'], mode='lines+markers',
            line=dict(width=f['width'], color=f"rgba({r},{g},{b},0.95)"),
            marker=dict(symbol='arrow', size=sizes, angleref='previous',
                        color=f"rgba({r},{g},{b},1)"),
            hoverinfo='text', text=f['hover'], showlegend=False,
        ))

    # ── Country anchors ─────────────────────────────────────────────────────
    exp_df = (
        df.groupby('reporterDesc')
        .agg(lon=('source_lon', 'first'), lat=('source_lat', 'first'),
             color=(color_col, 'first'), total=('primaryValue', 'sum'))
        .reset_index()
    )
    imp_df = (
        df.groupby('partnerDesc')
        .agg(lon=('target_lon', 'first'), lat=('target_lat', 'first'),
             total=('primaryValue', 'sum'))
        .reset_index()
    )
    imp_df = imp_df[~imp_df['partnerDesc'].isin(set(exp_df['reporterDesc']))]

    fig.add_trace(go.Scattergeo(                               # importers
        lon=imp_df['lon'], lat=imp_df['lat'],
        mode='markers+text',
        marker=dict(size=7, color='white',
                    line=dict(width=1.2, color='#334155')),
        text=imp_df['partnerDesc'], textposition='top center',
        textfont=dict(size=10, color='#475569', family='Arial'),
        hoverinfo='text',
        hovertext=[f"{r.partnerDesc} (importer)<br><b>{fmt(r.total)}</b> received"
                   for r in imp_df.itertuples()],
        showlegend=False,
    ))
    fig.add_trace(go.Scattergeo(                               # exporters
        lon=exp_df['lon'], lat=exp_df['lat'],
        mode='markers+text',
        marker=dict(size=13,
                    color=[f"rgb({c[0]},{c[1]},{c[2]})" for c in exp_df['color']],
                    line=dict(width=1.6, color='#0F172A')),
        text=exp_df['reporterDesc'], textposition='top center',
        textfont=dict(size=12, color='#0F172A', family='Arial Black, Arial'),
        hoverinfo='text',
        hovertext=[f"{r.reporterDesc} (exporter)<br><b>{fmt(r.total)}</b> total"
                   for r in exp_df.itertuples()],
        showlegend=False,
    ))

    # ── Shared geo styling — identical for both projections ────────────────
    proj = (dict(type='orthographic', rotation=dict(lon=115, lat=15, roll=0))
            if globe else dict(type='natural earth'))
    fig.update_layout(
        geo=dict(
            projection=proj,
            showland=True,      landcolor=LAND_HEX,
            showocean=True,     oceancolor=OCEAN_HEX,
            showcountries=True, countrycolor=BORDER_HEX, countrywidth=0.6,
            showcoastlines=False, showlakes=False, showrivers=False,
            showframe=False,
            bgcolor=FRAME_HEX,
        ),
        paper_bgcolor=FRAME_HEX,
        margin=dict(l=0, r=0, t=0, b=0),
        height=height,
        showlegend=False,
        hoverlabel=dict(bgcolor='white', font=dict(color='#0F172A')),
        # Keep the user's rotation/zoom when widgets trigger a re-render
        uirevision='flow-map',
    )
    return fig


_YLGNBU_DEST = ['#225ea8','#1d91c0','#41b6c4','#7fcdbb','#c7e9b4','#edf8b1','#ffffd9','#f7fcb9']

@st.cache_data
def build_sankey_fig(df_flow, hex_palette):
    """Two-column Sankey: exporters (left) → importers (right).

    A country may legitimately appear on BOTH sides — e.g. China is both an
    IC exporter and the world's largest IC importer; Korea both exports and
    imports equipment. Source and target nodes therefore live in SEPARATE
    index spaces (Plotly allows duplicate labels as long as indices differ).
    A single shared name→index dict would silently collapse the two roles
    and mis-wire the links."""
    # Guard against degenerate self-flows (Comtrade should never report a
    # country as its own partner, but the inner logic assumes it doesn't).
    df_flow = df_flow[df_flow['reporterDesc'] != df_flow['partnerDesc']]

    src_nodes = list(df_flow['reporterDesc'].unique())
    tgt_nodes = list(df_flow['partnerDesc'].unique())
    all_nodes = src_nodes + tgt_nodes
    src_idx = {n: i for i, n in enumerate(src_nodes)}
    tgt_idx = {n: i + len(src_nodes) for i, n in enumerate(tgt_nodes)}

    # Destination node colours: use the canonical country palette for any
    # country we recognise (so China is always red, Korea always teal, etc.
    # across BOTH the IC and Equipment Sankeys), then fall back to the
    # YlGnBu sequential palette for all-other importers.
    _fb_idx = 0
    tgt_colors = []
    for t in tgt_nodes:
        if t in ALL_COUNTRY_HEX:
            tgt_colors.append(ALL_COUNTRY_HEX[t])
        else:
            tgt_colors.append(_YLGNBU_DEST[_fb_idx % len(_YLGNBU_DEST)])
            _fb_idx += 1

    node_colors = (
        [hex_palette.get(n, '#888888') for n in src_nodes] +
        tgt_colors
    )
    link_sources = [src_idx[r] for r in df_flow['reporterDesc']]
    link_targets  = [tgt_idx[t] for t in df_flow['partnerDesc']]
    link_values   = (df_flow['primaryValue'] / 1e9).round(1).tolist()
    link_colors   = [hex_to_rgba(hex_palette.get(r, '#888888')) for r in df_flow['reporterDesc']]
    height = max(480, max(len(src_nodes), len(tgt_nodes)) * 60 + 100)
    fig = go.Figure(go.Sankey(
        node=dict(
            pad=20, thickness=20,
            label=all_nodes, color=node_colors,
            hovertemplate='%{label}<br>$%{value:.1f}B<extra></extra>',
        ),
        link=dict(
            source=link_sources, target=link_targets,
            value=link_values,   color=link_colors,
            hovertemplate='%{source.label} → %{target.label}<br>$%{value:.1f}B<extra></extra>',
        ),
        textfont=dict(size=13, color='#1a1a1a', family='sans-serif'),
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=10, r=10),
        height=height,
    )
    return fig

@st.cache_data
def build_ranking_bar_fig(ranked_df, title):
    """Horizontal bar, top-N exporters by cumulative share. Colour is a
    continuous gradient (YlGnBu) keyed on share_pct itself, matching the
    notebook's top10_composition_chart — darker/bluer bars simply mean a
    bigger share, independent of which country it is."""
    d = ranked_df.sort_values('share_pct', ascending=True)   # ascending: Plotly draws bottom-up
    fig = go.Figure(go.Bar(
        x=d['share_pct'], y=d['reporterDesc'], orientation='h',
        marker=dict(color=d['share_pct'], colorscale='YlGnBu', showscale=False),
        text=d['share_pct'].round(1).astype(str) + '%',
        textposition='outside',
        customdata=d['value_usd_bn'].round(1),
        hovertemplate='%{y}<br>%{customdata:.1f}B, %{x:.1f}% share<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        xaxis_title='Share of world exports, 2018–2024 (%)',
        plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(gridcolor='rgba(0,0,0,0.08)'),
        margin=dict(t=50, l=10, r=40, b=10),
        height=max(320, len(d) * 34),
    )
    return fig

# ── Load all data ──────────────────────────────────────────────────────────
with st.spinner("Loading UN Comtrade data. This may take a moment on first load..."):
    df_global       = load_global_data()
    df_global_equip = load_global_equip_data()
    df_reexport     = load_reexport_data()   # IC exports from intermediate hubs (HKG, SGP, MYS, VNM)
    ic_ranking      = rank_top_exporters(load_exporter_ranking('8542'))
    equip_ranking   = rank_top_exporters(load_exporter_ranking('8486'))

# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.title("Controls")

st.sidebar.markdown("---")
st.sidebar.markdown("### Select Year")
st.sidebar.caption("Applies to the global map and all tabs.")
year = st.sidebar.slider("Year", 2018, 2025, 2024, label_visibility="collapsed")

projection = st.sidebar.radio(
    "Projection",
    ["🌐 3D Globe", "🗺️ Flat Map"],
    index=0,
    help="The globe is continuous, so trans-Pacific flows are never cut off. Drag to rotate.",
)
use_globe = projection.endswith("Globe")

st.sidebar.markdown("---")
st.sidebar.markdown("### IC Exporters")
st.sidebar.caption(
    "Toggles IC export flows (HS 8542) from top IC exporters to importers. "
)
selected_countries = []
for country in IC_EXPORTERS:          # alphabetical: China, Japan, Korea, Taiwan, USA
    hex_c = src_hex[country]
    default_on = country in ('Taiwan', 'Rep. of Korea')
    if st.sidebar.checkbox(country, value=default_on, key=f"toggle_{country}"):
        selected_countries.append(country)

st.sidebar.markdown("### IC Re-exports Hubs")
st.sidebar.caption(
    "Overlay IC re-export flows (HS 8542) from key intermediate hubs onto the IC map. "
    "These countries receive chips from primary exporters and forward them to final "
    "assembly markets, revealing the second hop of the supply chain."
)
selected_hubs = []
for hub in REEXPORT_HUBS:             # alphabetical: Hong Kong, Malaysia, Singapore, Vietnam
    colour_swatch = hub_hex[hub]
    if st.sidebar.checkbox(hub, value=False, key=f"hub_{hub}"):
        selected_hubs.append(hub)

st.sidebar.markdown("---")
st.sidebar.markdown("### Equipment Exporters")
st.sidebar.caption("Applies to the Equipment section of Tab 1.")
selected_equip_countries = []
for country in EQUIP_EXPORTERS:       # alphabetical: Germany, Japan, Netherlands, Korea, USA
    hex_c = equip_src_hex[country]
    if st.sidebar.checkbox(country, value=(country == 'Netherlands'), key=f"eq_toggle_{country}"):
        selected_equip_countries.append(country)

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Source:** UN Comtrade  \n"
    "**Products:** HS 8486 (equipment), HS 8542 (ICs)  \n"
    "**Coverage:** 2018–2025  \n"
    "**Flow:** Gross exports (reporter-side, HS classification)"
)

@st.cache_data
def china_bound_totals(df, exporters, year_a, year_b):
    """Sum of exports FROM the given exporters TO China only (not their
    total world exports), for two specific years. This is the number the
    top-of-page narrative needs — 'exports to China' — which is different
    from both df_global's total-exports-by-country and the china_share
    metric (China's own exports as a % of tracked total)."""
    d = df[
        (df['reporterDesc'].isin(exporters)) &
        (df['reporterDesc'] != 'China') &      # exclude China as its own destination
        (df['partnerDesc'] == 'China')
    ]
    by_year = d.groupby('period')['primaryValue'].sum()
    val_a = by_year.get(str(year_a), 0.0)
    val_b = by_year.get(str(year_b), 0.0)
    pct_change = ((val_b - val_a) / val_a * 100) if val_a else None
    return val_a, val_b, pct_change

st.title("Global Semiconductor Trade Flows")
st.markdown(
    "This dashboard pulls data from the UN Comtrade to analyse semiconductor trade data up to 2024. "
    "While datasets up till 2025 exist, several key countries such as China and Taiwan have yet to "
    "report. Hence, the dataset is only as coherent up till 2024. Nevertheless, the story it tells is "
    "illuminating. Since the US has imposed the Export Controls on Advanced Computing and Semiconductors "
    "in October 2022, IC and Semiconductor Equipment exports to China has slowed."
)

st.divider()

# ══ ACT 1 — CONCENTRATION ═══════════════════════════════════════════════
st.header("🔬 Integrated Circuits (HS 8542)")
st.caption(
    "Exports from major IC-exporting nations. "
    "Left = exporters, right = importers. Data: UN Comtrade."
)

st.plotly_chart(
    build_ranking_bar_fig(ic_ranking, "Top 15 IC (HS 8542) Exporters, 2018–2024"),
    width='stretch',
)

df_arcs = build_arc_df(df_global)
# All partner flows are kept — no top-N truncation (small flows through
# re-export hubs like Singapore or Malaysia matter to the routing story).

df_year     = df_arcs[(df_arcs['period']==str(year)) & (df_arcs['reporterDesc'].isin(selected_countries))].copy()

# ── Map ↔ Sankey consistency ────────────────────────────────────────────
# The map keeps the YEAR's top-15 partners (beyond that, marginal utility
# is low and the map clutters), UNIONED with the destinations the Sankey
# below will show for the same year — so any flow visible in the Sankey
# (e.g. Netherlands → Singapore on the equipment side) is guaranteed to
# also appear on the map. Both views now include producer nations as
# destinations, so map and Sankey share the same inclusion philosophy.
# NOTE: producer nations are deliberately INCLUDED as destinations.
# China is the world's largest IC importer; excluding exporters from the
# destination side (as earlier versions did) hid the dominant flows of
# the entire system (Taiwan→China, Korea→China) and showed only the
# residual periphery.
sankey_dest_ic = set(
    df_global[(df_global['period'] == str(year)) &
              (df_global['reporterDesc'].isin(selected_countries))]
    .groupby('partnerDesc')['primaryValue'].sum().nlargest(15).index
)
top15_ic = set(
    df_year.groupby('partnerISO')['primaryValue'].sum().nlargest(15).index
)
df_year = df_year[
    df_year['partnerISO'].isin(top15_ic) |
    df_year['partnerDesc'].isin(sankey_dest_ic)
].copy()

# Diagnostic: partners that can never appear on the map because they have
# no entry in the coordinates table (the coords merge is an inner join).
_ic_known = set(coords_df['ISO'])
_ic_missing = (
    df_global[(df_global['period'] == str(year)) &
              (df_global['reporterDesc'].isin(selected_countries)) &
              (~df_global['partnerISO'].isin(_ic_known))]
    .groupby('partnerDesc')['primaryValue'].sum().nlargest(5)
)
if not _ic_missing.empty and _ic_missing.iloc[0] > 1e9:
    st.caption(
        "⚠️ Not mappable (no coordinates on file): "
        + ", ".join(f"{k} ({fmt(v)})" for k, v in _ic_missing.items())
    )
df_year_all = df_global[df_global['period']==str(year)]
df_prev_all = df_global[df_global['period']==str(year-1)] if year > 2018 else None

total_cur    = df_year_all['primaryValue'].sum()
total_prev   = df_prev_all['primaryValue'].sum() if df_prev_all is not None else None
taiwan_share = df_year_all[df_year_all['reporterDesc']=='Taiwan']['primaryValue'].sum() / total_cur * 100
china_share  = df_year_all[df_year_all['reporterDesc']=='China']['primaryValue'].sum()  / total_cur * 100
yoy          = f"{((total_cur-total_prev)/total_prev*100):+.1f}% YoY" if total_prev else None

c1, c2, c3 = st.columns(3)
c1.metric("Tracked IC Exports (5 reporters)",  fmt(total_cur), delta=yoy,
          help="Sum of gross HS 8542 exports from the five tracked reporter "
               "nations only — NOT world exports. Gross flows double-count "
               "chips that cross borders more than once.")
c2.metric("Taiwan's Share of Tracked",  f"{taiwan_share:.1f}%",
          help="Taiwan ÷ sum of the five tracked reporters. Taiwan's share "
               "of *world* IC exports is lower.")
c3.metric("China's Share of Tracked",   f"{china_share:.1f}%",
          help="China ÷ sum of the five tracked reporters.")
st.markdown("---")

if selected_countries and not df_year.empty:
    df_year['color'] = df_year['reporterDesc'].map(country_rgba)
    # Square-root scaling keeps large flows readable without drowning small
    # ones. Normalised against the ALL-YEARS maximum so ribbon widths are
    # comparable when scrubbing the year slider (a $50B flow looks the same
    # in 2019 and 2025).
    global_max_ic = df_arcs['primaryValue'].max()
    df_year['width'] = (np.sqrt(df_year['primaryValue'] / global_max_ic) * 6).clip(lower=1.0)
    df_year['value_fmt'] = df_year['primaryValue'].apply(fmt)

    # ── Re-export hub overlay ─────────────────────────────────────────
    # When hubs are toggled on, build their outbound arc dataset and
    # layer it on top of the primary exporter flows.  Same global_max_ic
    # normalisation ensures hub ribbons are visually proportional
    # (hub volumes are smaller, so arcs naturally appear thinner).
    df_hub_ic = pd.DataFrame()
    if selected_hubs and not df_reexport.empty:
        _hub_arcs = build_arc_df(df_reexport)
        _hub_yr = _hub_arcs[
            (_hub_arcs['period'] == str(year)) &
            (_hub_arcs['reporterDesc'].isin(selected_hubs))
        ].copy()
        if not _hub_yr.empty:
            # Cap at top-8 destinations per hub combined to keep the map legible
            _top_dest = set(
                _hub_yr.groupby('partnerISO')['primaryValue'].sum().nlargest(8).index
            )
            _hub_yr = _hub_yr[_hub_yr['partnerISO'].isin(_top_dest)]
            _hub_yr['color']     = _hub_yr['reporterDesc'].map(hub_rgba)
            _hub_yr['width']     = (np.sqrt(_hub_yr['primaryValue'] / global_max_ic) * 6).clip(lower=1.0)
            _hub_yr['value_fmt'] = _hub_yr['primaryValue'].apply(fmt)
            df_hub_ic = _hub_yr

    # Merge primary and hub flows into a single figure
    df_ic_plot = (
        pd.concat([df_year, df_hub_ic], ignore_index=True)
        if not df_hub_ic.empty else df_year
    )

    _hub_key = '_'.join(sorted(selected_hubs)) if selected_hubs else 'none'
    st.plotly_chart(
        build_flow_fig(df_ic_plot, globe=use_globe),
        key=f"arc_map_{year}_{projection}_{'_'.join(sorted(selected_countries))}_{_hub_key}",
        width='stretch',
    )
    _hub_note = (
        " **Hub re-export arcs** (amber/pink/green/purple) show IC exports from "
        "the selected intermediate hubs to their top destinations."
        if selected_hubs else ""
    )
    st.caption(
        "**How to read this map** — Each line is an export flow; the arrowhead sits at the "
        "importer. Width ∝ √(trade value), colour = exporting country. "
        + ("Drag to rotate the globe; " if use_globe else "")
        + "hover any flow or dot for exact values."
        + _hub_note
    )
else:
    st.info("Select at least one IC exporter in the sidebar to display the map.")

st.divider()

st.caption(
    f"Left: {len(IC_EXPORTERS)} major IC-exporting nations. "
    "Right: their top 15 import destinations for the selected year. "
    "Producer nations appear on both sides — China, Korea, Taiwan, the USA and Japan "
    "are simultaneously among the largest IC exporters *and* importers, because chips "
    "cross borders repeatedly between fabrication, test/assembly and final integration. "
    "**China is consistently the single largest import destination** — the concentration "
    "story isn't just who makes chips, it's how dependent the system is on one buyer. "
    "Values are gross trade flows, not value-added."
)

df_sk = (
    df_global[df_global['period'] == str(year)]
    .groupby(['reporterDesc', 'partnerDesc'])['primaryValue']
    .sum().reset_index()
)
df_sk = df_sk[df_sk['reporterDesc'].isin(selected_countries)]
top_dest = sorted(sankey_dest_ic)   # same set the map was guaranteed to include
df_sk    = df_sk[df_sk['partnerDesc'].isin(top_dest)]

if not df_sk.empty:
    st.plotly_chart(build_sankey_fig(df_sk, src_hex), width='stretch')
else:
    st.info("Select at least one IC exporter in the sidebar to display the Sankey.")

##  IC Export Trends by Country
st.divider()

df_trend = (
    df_global[df_global['reporterDesc'].isin(selected_countries)]
    .groupby(['reporterDesc','period'])['primaryValue'].sum().reset_index()
)
df_trend['period']  = df_trend['period'].astype(int)
df_trend['value_B'] = df_trend['primaryValue'] / 1e9
fig_ts = px.line(
    df_trend, x='period', y='value_B', color='reporterDesc', markers=True,
    labels={'value_B':'Export Value (USD Billion)','period':'Year','reporterDesc':'Country'},
    color_discrete_map=src_hex,
)
fig_ts.add_vline(x=year, line_dash='dot',  line_color='#888888', opacity=0.5)
fig_ts.add_vline(x=2022, line_dash='dash', line_color='#DC2626', opacity=0.9)
if not df_trend.empty:
    fig_ts.add_annotation(
        x=2022.05, y=df_trend['value_B'].max()*0.95,
        text="US Export Controls<br>Oct 2022", showarrow=False,
        font=dict(color='#DC2626', size=11), xanchor='left'
    )
fig_ts.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    legend_title='Country', hovermode='x unified',
    xaxis=dict(tickmode='linear', dtick=1, gridcolor='rgba(0,0,0,0.08)'),
    yaxis=dict(gridcolor='rgba(0,0,0,0.08)'), margin=dict(t=20),
)
st.plotly_chart(fig_ts, width='stretch')

# ══ SEMICONDUCTOR EQUIPMENT (HS 8486) ═════════════════
st.divider()
st.header("⚙️ Semiconductor Equipment (HS 8486)")
st.caption(
    "Exports of lithography machines (EUV/DUV), etch tools, deposition systems, and metrology. "
    "Left = exporters, right = importers. Data: UN Comtrade."
)

st.plotly_chart(
    build_ranking_bar_fig(equip_ranking, "Top 15 Equipment (HS 8486) Exporters, 2018–2024"),
    width='stretch',
)

st.markdown(
    "> **Why these countries?** These five nations control the critical tooling that every chip fab depends on. "
    "**Netherlands** (ASML, sole maker of EUV lithography), "
    "**USA** (Applied Materials, Lam Research, KLA), "
    "**Japan** (Tokyo Electron, Nikon, Canon, Advantest), "
    "**Germany** (Carl Zeiss, whose optics are inside every ASML machine; Aixtron), "
    "**South Korea** (Samsung's equipment division, Jusung Engineering)."
)

df_arcs_eq = build_arc_df(df_global_equip)

df_eq_year     = df_arcs_eq[(df_arcs_eq['period']==str(year)) & (df_arcs_eq['reporterDesc'].isin(selected_equip_countries))].copy()

# Map ↔ Sankey consistency (see IC section for rationale).
# Producer nations included as destinations: Korea and the USA are both
# major equipment exporters and major equipment importers.
sankey_dest_eq = set(
    df_global_equip[(df_global_equip['period'] == str(year)) &
                    (df_global_equip['reporterDesc'].isin(selected_equip_countries))]
    .groupby('partnerDesc')['primaryValue'].sum().nlargest(15).index
)
top15_eq = set(
    df_eq_year.groupby('partnerISO')['primaryValue'].sum().nlargest(15).index
)
df_eq_year = df_eq_year[
    df_eq_year['partnerISO'].isin(top15_eq) |
    df_eq_year['partnerDesc'].isin(sankey_dest_eq)
].copy()

_eq_missing = (
    df_global_equip[(df_global_equip['period'] == str(year)) &
                    (df_global_equip['reporterDesc'].isin(selected_equip_countries)) &
                    (~df_global_equip['partnerISO'].isin(set(coords_df['ISO'])))]
    .groupby('partnerDesc')['primaryValue'].sum().nlargest(5)
)
if not _eq_missing.empty and _eq_missing.iloc[0] > 1e9:
    st.caption(
        "⚠️ Not mappable (no coordinates on file): "
        + ", ".join(f"{k} ({fmt(v)})" for k, v in _eq_missing.items())
    )
df_eq_year_all = df_global_equip[df_global_equip['period']==str(year)]
df_eq_prev_all = df_global_equip[df_global_equip['period']==str(year-1)] if year > 2018 else None

total_eq_cur  = df_eq_year_all['primaryValue'].sum()
total_eq_prev = df_eq_prev_all['primaryValue'].sum() if df_eq_prev_all is not None else None
yoy_eq        = f"{((total_eq_cur-total_eq_prev)/total_eq_prev*100):+.1f}% YoY" if total_eq_prev else None
nld_share     = (
    df_eq_year_all[df_eq_year_all['reporterDesc']=='Netherlands']['primaryValue'].sum()
    / total_eq_cur * 100
) if total_eq_cur > 0 else 0
producer_set_eq = set(EQUIP_EXPORTERS)
# Largest buyer computed over ALL destinations, including producer
# nations — Korea and Taiwan are among the world's biggest tool buyers,
# and excluding them (as earlier versions did) misstated the answer.
top_eq_dest_ser = (
    df_eq_year_all
    .groupby('partnerDesc')['primaryValue'].sum()
)
top_eq_importer = top_eq_dest_ser.idxmax() if not top_eq_dest_ser.empty else 'N/A'

e1, e2, e3 = st.columns(3)
e1.metric("Tracked Equipment Exports (5 reporters)", fmt(total_eq_cur), delta=yoy_eq,
          help="Sum of gross HS 8486 exports from the five tracked reporter "
               "nations only — NOT world exports. Note HS 8486 also covers "
               "flat-panel-display equipment, and excludes test equipment "
               "(HS 9030), so this is an imperfect proxy for 'semiconductor "
               "equipment'.")
e2.metric("Netherlands' Share of Tracked", f"{nld_share:.1f}%",
          help="Netherlands ÷ sum of the five tracked reporters. Dutch HS 8486 "
               "exports are dominated by ASML but also include ASM International "
               "and BESI.")
e3.metric("Largest Equipment Buyer", top_eq_importer,
          help="Largest import destination among the tracked reporters' exports, "
               "including producer nations themselves.")
st.markdown("---")

if selected_equip_countries and not df_eq_year.empty:
    df_eq_year['color'] = df_eq_year['reporterDesc'].map(country_rgba)
    global_max_eq = df_arcs_eq['primaryValue'].max()
    df_eq_year['width'] = (np.sqrt(df_eq_year['primaryValue'] / global_max_eq) * 6).clip(lower=1.0)
    df_eq_year['value_fmt'] = df_eq_year['primaryValue'].apply(fmt)
    st.plotly_chart(
        build_flow_fig(df_eq_year, globe=use_globe),
        key=f"arc_eq_{year}_{projection}_{'_'.join(sorted(selected_equip_countries))}",
        width='stretch',
    )
    st.caption(
        "**How to read this map** — Each line is an export flow; the arrowhead sits at the "
        "importer. Width ∝ √(trade value), colour = exporting country."
    )
else:
    st.info("Select at least one equipment exporter in the sidebar to display the map.")

st.markdown("---")
st.subheader("Equipment exporters to importers")
st.caption(
    "Left: 5 major equipment-exporting nations. "
    "Right: their top 15 import destinations for the selected year. "
    "Producer nations appear on both sides — Korea and the USA both export and import "
    "tools, and intra-supply-chain shipments (e.g. Zeiss optics from Germany to ASML "
    "in the Netherlands) are real flows, not noise. Values are gross trade flows."
)

df_sk_eq = (
    df_global_equip[df_global_equip['period'] == str(year)]
    .groupby(['reporterDesc', 'partnerDesc'])['primaryValue']
    .sum().reset_index()
)
df_sk_eq = df_sk_eq[df_sk_eq['reporterDesc'].isin(selected_equip_countries)]
top_eq_dest = sorted(sankey_dest_eq)   # same set the map was guaranteed to include
df_sk_eq    = df_sk_eq[df_sk_eq['partnerDesc'].isin(top_eq_dest)]

if not df_sk_eq.empty:
    st.plotly_chart(build_sankey_fig(df_sk_eq, equip_src_hex), width='stretch')
else:
    st.info("Select at least one equipment exporter in the sidebar to display the Sankey.")

st.markdown("---")
st.subheader("Equipment Export Trends by Country (2018–2025)")
df_eq_trend_global = (
    df_global_equip[df_global_equip['reporterDesc'].isin(selected_equip_countries)]
    .groupby(['reporterDesc','period'])['primaryValue'].sum().reset_index()
)
df_eq_trend_global['period']  = df_eq_trend_global['period'].astype(int)
df_eq_trend_global['value_B'] = df_eq_trend_global['primaryValue'] / 1e9
fig_eq_ts = px.line(
    df_eq_trend_global, x='period', y='value_B', color='reporterDesc', markers=True,
    labels={'value_B':'Export Value (USD Billion)','period':'Year','reporterDesc':'Country'},
    color_discrete_map=equip_src_hex,
)
fig_eq_ts.add_vline(x=year, line_dash='dot',  line_color='#888888', opacity=0.5)
fig_eq_ts.add_vline(x=2022, line_dash='dash', line_color='#DC2626', opacity=0.9)
if not df_eq_trend_global.empty:
    fig_eq_ts.add_annotation(
        x=2022.05, y=df_eq_trend_global['value_B'].max()*0.95,
        text="US Export Controls<br>Oct 2022", showarrow=False,
        font=dict(color='#DC2626', size=11), xanchor='left'
    )
fig_eq_ts.update_layout(
    plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
    legend_title='Country', hovermode='x unified',
    xaxis=dict(tickmode='linear', dtick=1, gridcolor='rgba(0,0,0,0.08)'),
    yaxis=dict(gridcolor='rgba(0,0,0,0.08)'), margin=dict(t=20),
)
st.plotly_chart(fig_eq_ts, width='stretch')
