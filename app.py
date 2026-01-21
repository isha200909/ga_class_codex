from pathlib import Path
import re
import subprocess
import sys

import pandas as pd
import streamlit as st

try:
    from wordcloud import WordCloud, STOPWORDS
except ImportError:
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wordcloud"])
        from wordcloud import WordCloud, STOPWORDS
    except Exception:
        WordCloud = None
        STOPWORDS = set()

DATA_PATH = Path(__file__).parent / "data" / "singapore_airlines_reviews.csv"

st.set_page_config(page_title="SIA Review Pulse", page_icon="??", layout="wide")


@st.cache_data
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df.columns = [c.replace("\ufeff", "").replace('"', "").strip() for c in df.columns]

    if "published_date" in df.columns:
        df["published_date"] = pd.to_datetime(df["published_date"], errors="coerce", utc=True)
        df["published_date"] = df["published_date"].dt.tz_convert(None)
    if "rating" in df.columns:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    if "helpful_votes" in df.columns:
        df["helpful_votes"] = pd.to_numeric(df["helpful_votes"], errors="coerce")

    if "published_date" in df.columns:
        df["year_month"] = df["published_date"].dt.to_period("M").dt.to_timestamp()

    return df


df = load_data(DATA_PATH)

st.title("SIA Review Pulse")
summary_placeholder = st.empty()
st.subheader("Singapore Airlines review insights")

with st.sidebar:
    st.header("Filters")

    if "published_date" in df.columns and df["published_date"].notna().any():
        min_date = df["published_date"].min().date()
        max_date = df["published_date"].max().date()
        start_date = st.date_input(
            "Start date",
            value=min_date,
            min_value=min_date,
            max_value=max_date,
        )
        end_date = st.date_input(
            "End date",
            value=max_date,
            min_value=min_date,
            max_value=max_date,
        )
    else:
        start_date, end_date = None, None

    platform_options = sorted(df["published_platform"].dropna().unique()) if "published_platform" in df.columns else []
    selected_platforms = st.multiselect("Platform", platform_options, default=platform_options)

    rating_min = int(df["rating"].min()) if "rating" in df.columns and df["rating"].notna().any() else 1
    rating_max = int(df["rating"].max()) if "rating" in df.columns and df["rating"].notna().any() else 5
    selected_rating = st.slider("Rating range", rating_min, rating_max, (rating_min, rating_max))

    keyword_options = ["food", "seat", "crew", "delay", "service", "lounge", "baggage", "check-in", "entertainment"]
    selected_keywords = st.multiselect("Keyword contains", keyword_options)

    search_query = st.text_input("Search title or text", value="").strip()


filtered = df.copy()

if start_date and end_date and "published_date" in filtered.columns:
    start_date, end_date = sorted([start_date, end_date])
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    filtered = filtered[(filtered["published_date"] >= start_ts) & (filtered["published_date"] <= end_ts)]

if selected_platforms and "published_platform" in filtered.columns:
    filtered = filtered[filtered["published_platform"].isin(selected_platforms)]

if selected_keywords:
    text_cols = []
    if "title" in filtered.columns:
        text_cols.append(filtered["title"].fillna(""))
    if "text" in filtered.columns:
        text_cols.append(filtered["text"].fillna(""))
    if text_cols:
        combined = text_cols[0].astype(str)
        for col in text_cols[1:]:
            combined = combined + " " + col.astype(str)
        keyword_pattern = "|".join([re.escape(k) for k in selected_keywords])
        filtered = filtered[combined.str.contains(keyword_pattern, case=False, na=False)]

if "rating" in filtered.columns:
    filtered = filtered[filtered["rating"].between(selected_rating[0], selected_rating[1])]

if search_query:
    text_cols = []
    if "title" in filtered.columns:
        text_cols.append(filtered["title"].fillna(""))
    if "text" in filtered.columns:
        text_cols.append(filtered["text"].fillna(""))
    if text_cols:
        combined = text_cols[0].astype(str)
        for col in text_cols[1:]:
            combined = combined + " " + col.astype(str)
        filtered = filtered[combined.str.contains(search_query, case=False, na=False)]

if filtered.empty:
    st.warning("No reviews match the selected filters.")
    st.stop()

total_reviews = len(filtered)
avg_rating = filtered["rating"].mean() if "rating" in filtered.columns else None
positive_pct = (filtered["rating"].between(4, 5).mean() * 100) if "rating" in filtered.columns else None
negative_pct = (filtered["rating"].between(1, 2).mean() * 100) if "rating" in filtered.columns else None

summary_parts = [f"Total reviews: {total_reviews:,}"]
if avg_rating is not None:
    summary_parts.append(f"Average rating: {avg_rating:.2f}")
else:
    summary_parts.append("Average rating: n/a")
if positive_pct is not None:
    summary_parts.append(f"Positive reviews (ratings 4-5): {positive_pct:.1f}%")
else:
    summary_parts.append("Positive reviews (ratings 4-5): n/a")
if negative_pct is not None:
    summary_parts.append(f"Negative reviews (ratings 1-2): {negative_pct:.1f}%")
else:
    summary_parts.append("Negative reviews (ratings 1-2): n/a")

summary_placeholder.markdown(" | ".join(summary_parts))

metric_cols = st.columns(4)
metric_cols[0].metric("Total reviews", f"{len(filtered):,}")
if "rating" in filtered.columns:
    metric_cols[1].metric("Average rating", f"{filtered['rating'].mean():.2f}")
else:
    metric_cols[1].metric("Average rating", "n/a")

if "rating" in filtered.columns:
    positive_share = (filtered["rating"] >= 4).mean() * 100
    metric_cols[2].metric("Positive reviews", f"{positive_share:.1f}%")
else:
    metric_cols[2].metric("Positive reviews", "n/a")

if "helpful_votes" in filtered.columns:
    metric_cols[3].metric("Helpful votes", f"{int(filtered['helpful_votes'].sum()):,}")
else:
    metric_cols[3].metric("Helpful votes", "n/a")

st.markdown("---")

chart_cols = st.columns(2)

if "rating" in filtered.columns:
    rating_counts = (
        filtered["rating"]
        .dropna()
        .astype(int)
        .value_counts()
        .sort_index()
        .rename("count")
    )
    chart_cols[0].subheader("Ratings distribution")
    chart_cols[0].bar_chart(rating_counts)

if "year_month" in filtered.columns:
    monthly_counts = filtered.groupby("year_month").size().rename("reviews")
    chart_cols[1].subheader("Reviews over time")
    chart_cols[1].line_chart(monthly_counts)

if "published_platform" in filtered.columns:
    st.subheader("Reviews by platform")
    platform_counts = filtered["published_platform"].value_counts().rename("reviews")
    st.bar_chart(platform_counts)

st.subheader("Latest matched reviews")
show_cols = [c for c in ["published_date", "title", "rating", "published_platform", "helpful_votes"] if c in filtered.columns]
latest = filtered.sort_values("published_date", ascending=False) if "published_date" in filtered.columns else filtered
if selected_keywords or search_query:
    st.dataframe(latest[show_cols].head(20), use_container_width=True)
else:
    st.info("Enter a search query or select keywords to see matching latest reviews.")

st.markdown("---")
st.subheader("Keyword clouds")


def build_cloud_text(frame: pd.DataFrame) -> str:
    text_cols = []
    if "title" in frame.columns:
        text_cols.append(frame["title"].fillna(""))
    if "text" in frame.columns:
        text_cols.append(frame["text"].fillna(""))
    if not text_cols:
        return ""
    combined = text_cols[0].astype(str)
    for col in text_cols[1:]:
        combined = combined + " " + col.astype(str)
    return " ".join(combined.tolist())


def render_wordcloud(text: str, caption: str) -> None:
    if not text.strip():
        st.info(f"No {caption.lower()} reviews available for a keyword cloud.")
        return
    if WordCloud is None:
        st.error("wordcloud is not available. Please install it to render keyword clouds.")
        return
    stopwords = set(STOPWORDS)
    wc = WordCloud(
        width=1000,
        height=600,
        background_color="white",
        stopwords=stopwords,
        collocations=False,
    ).generate(text)
    st.image(wc.to_array(), use_container_width=True)


cloud_cols = st.columns(2)

if "rating" in filtered.columns:
    positive = filtered[filtered["rating"].between(4, 5)]
    negative = filtered[filtered["rating"].between(1, 2)]
else:
    positive = filtered.iloc[0:0]
    negative = filtered.iloc[0:0]

with cloud_cols[0]:
    st.markdown("**Positive reviews (ratings 4-5)**")
    render_wordcloud(build_cloud_text(positive), "Positive")

with cloud_cols[1]:
    st.markdown("**Negative reviews (ratings 1-2)**")
    render_wordcloud(build_cloud_text(negative), "Negative")
