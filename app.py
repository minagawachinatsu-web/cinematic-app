import io
from datetime import datetime, timezone
import colorsys

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from PIL import Image
from sklearn.cluster import KMeans


# ============================================================
# 設定
# ============================================================

st.set_page_config(
    page_title="映画カラー分析 | TMDb Top 5",
    page_icon="🎬",
    layout="wide",
)

TMDB_BASE_URL = "https://api.themoviedb.org/3"
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"

GENRES = {
    "アクション": 28,
    "ホラー": 27,
    "ヒューマンドラマ": 18,
    "恋愛": 10749,
    "アニメ": 16,
}


# ============================================================
# CSS
# ============================================================

st.markdown(
    """
    <style>
    .color-bar {
        height: 30px;
        border-radius: 6px;
        margin-top: 3px;
        margin-bottom: 2px;
    }

    .movie-card {
        border: 1px solid #dddddd;
        border-radius: 12px;
        padding: 12px;
        margin-bottom: 12px;
    }

    .small {
        font-size: 0.82rem;
        color: #666666;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# APIキー取得
# ============================================================

def get_secret_api_key():
    try:
        return st.secrets["TMDB_API_KEY"]
    except Exception:
        return ""


secret_api_key = get_secret_api_key()

st.sidebar.title("🎬 TMDb設定")

if secret_api_key:
    st.sidebar.success(
        "Streamlit SecretsからAPIキーを読み込みました。"
    )
    api_key = secret_api_key
else:
    api_key = st.sidebar.text_input(
        "TMDb APIキー",
        type="password",
        help="GitHubには保存しないでください。",
    )

st.sidebar.markdown("---")

st.sidebar.write(
    "「最新データ取得」を押すと、TMDbから"
    "各ジャンルの人気順Top 5を取得します。"
)

fetch_button = st.sidebar.button(
    "🔄 最新データ取得",
    type="primary",
    use_container_width=True,
)


# ============================================================
# TMDbからジャンル別Top 5を取得
# ============================================================

def fetch_top5_movies(api_key, genre_name, genre_id):

    url = f"{TMDB_BASE_URL}/discover/movie"

    params = {
        "api_key": api_key,
        "language": "ja-JP",
        "sort_by": "popularity.desc",
        "with_genres": genre_id,
        "include_adult": "false",
        "include_video": "false",
        "page": 1,
    }

    response = requests.get(
        url,
        params=params,
        timeout=20,
    )

    response.raise_for_status()

    results = response.json().get(
        "results",
        []
    )

    movies = []

    for movie in results[:5]:

        poster_path = movie.get(
            "poster_path"
        )

        if not poster_path:
            continue

        movies.append(
            {
                "genre": genre_name,
                "genre_id": genre_id,
                "tmdb_id": movie.get("id"),
                "title": movie.get(
                    "title",
                    "タイトル不明",
                ),
                "original_title": movie.get(
                    "original_title",
                    "",
                ),
                "year": (
                    movie.get(
                        "release_date",
                        "",
                    )[:4]
                    if movie.get("release_date")
                    else "----"
                ),
                "rating": movie.get(
                    "vote_average",
                    0,
                ),
                "vote_count": movie.get(
                    "vote_count",
                    0,
                ),
                "overview": movie.get(
                    "overview",
                    "概要情報がありません。",
                ),
                "poster_path": poster_path,
                "poster_url": (
                    TMDB_IMAGE_BASE_URL
                    + poster_path
                ),
            }
        )

    return movies


# ============================================================
# 画像取得
# ============================================================

def download_image(url):

    response = requests.get(
        url,
        timeout=20,
        headers={
            "User-Agent": "Mozilla/5.0"
        },
    )

    response.raise_for_status()

    return Image.open(
        io.BytesIO(
            response.content
        )
    ).convert("RGB")


# ============================================================
# K-meansによる主要5色抽出
# ============================================================

def extract_palette(
    image,
    n_colors=5,
):

    image = image.copy()

    image.thumbnail(
        (300, 300)
    )

    pixels = list(
        image.getdata()
    )

    step = max(
        1,
        len(pixels) // 4000
    )

    pixels = pixels[::step]

    model = KMeans(
        n_clusters=n_colors,
        random_state=42,
        n_init=10,
    )

    labels = model.fit_predict(
        pixels
    )

    centers = (
        model.cluster_centers_
        .astype(int)
    )

    counts = (
        pd.Series(labels)
        .value_counts()
    )

    total = len(labels)

    palette = []

    for index, center in enumerate(
        centers
    ):

        percentage = (
            counts.get(index, 0)
            / total
            * 100
        )

        hex_color = (
            "#{:02X}{:02X}{:02X}"
            .format(
                center[0],
                center[1],
                center[2],
            )
        )

        palette.append(
            {
                "hex": hex_color,
                "percentage": round(
                    percentage,
                    1,
                ),
            }
        )

    return sorted(
        palette,
        key=lambda x: x["percentage"],
        reverse=True,
    )


# ============================================================
# HEX → HSV
# ============================================================

def hex_to_hsv(hex_color):

    value = hex_color.lstrip("#")

    r = (
        int(value[0:2], 16)
        / 255
    )

    g = (
        int(value[2:4], 16)
        / 255
    )

    b = (
        int(value[4:6], 16)
        / 255
    )

    return colorsys.rgb_to_hsv(
        r,
        g,
        b,
    )


# ============================================================
# 明度・彩度計算
# ============================================================

def calculate_features(
    palette
):

    brightness = 0
    saturation = 0
    total = 0

    for color in palette:

        _, s, v = hex_to_hsv(
            color["hex"]
        )

        weight = color[
            "percentage"
        ]

        brightness += (
            v * weight
        )

        saturation += (
            s * weight
        )

        total += weight

    if total == 0:
        return 0, 0

    return (
        round(
            brightness
            / total
            * 100,
            1,
        ),
        round(
            saturation
            / total
            * 100,
            1,
        ),
    )


# ============================================================
# 全ジャンル取得＋カラー分析
# ============================================================

def fetch_and_analyze_all(
    api_key
):

    all_movies = []

    progress = st.progress(
        0,
        text="TMDbから作品情報を取得しています...",
    )

    total_genres = len(
        GENRES
    )

    for index, (
        genre_name,
        genre_id,
    ) in enumerate(
        GENRES.items(),
        start=1,
    ):

        progress.progress(
            (index - 1)
            / total_genres,
            text=(
                f"{genre_name}のTop 5を取得中..."
            ),
        )

        movies = fetch_top5_movies(
            api_key,
            genre_name,
            genre_id,
        )

        for movie in movies:

            try:

                image = download_image(
                    movie["poster_url"]
                )

                movie["palette"] = (
                    extract_palette(
                        image,
                        n_colors=5,
                    )
                )

                (
                    movie["brightness"],
                    movie["saturation"],
                ) = calculate_features(
                    movie["palette"]
                )

                movie[
                    "analysis_status"
                ] = "OK"

            except Exception as error:

                movie["palette"] = []

                movie[
                    "brightness"
                ] = None

                movie[
                    "saturation"
                ] = None

                movie[
                    "analysis_status"
                ] = (
                    f"画像分析失敗: {error}"
                )

            all_movies.append(
                movie
            )

    progress.progress(
        1.0,
        text="取得・カラー分析が完了しました。",
    )

    return all_movies


# ============================================================
# セッション状態
# ============================================================

if "movies" not in st.session_state:
    st.session_state.movies = []

if "retrieved_at" not in st.session_state:
    st.session_state.retrieved_at = None


# ============================================================
# 最新データ取得
# ============================================================

if fetch_button:

    if not api_key:

        st.error(
            "TMDb APIキーが入力されていません。"
        )

    else:

        try:

            with st.spinner(
                "25作品の取得とK-means分析を実行中..."
            ):

                movies = (
                    fetch_and_analyze_all(
                        api_key
                    )
                )

            st.session_state.movies = (
                movies
            )

            st.session_state.retrieved_at = (
                datetime.now(
                    timezone.utc
                )
                .astimezone()
                .strftime(
                    "%Y-%m-%d %H:%M:%S %Z"
                )
            )

            st.success(
                f"{len(movies)}作品の取得・分析が完了しました。"
            )

        except requests.HTTPError as error:

            st.error(
                "TMDb APIへのアクセスに失敗しました。"
                "APIキーを確認してください。"
            )

            st.code(
                str(error)
            )

        except Exception as error:

            st.error(
                "予期しないエラーが発生しました。"
            )

            st.exception(error)


movies = st.session_state.movies


# ============================================================
# タイトル
# ============================================================

st.title(
    "🎬 映画カラー分析"
)

st.subheader(
    "TMDb・各ジャンル人気Top 5 × ポスター色彩分析"
)

st.markdown(
    """
    このアプリは、TMDbのDiscover Movie APIを使い、
    各ジャンルを人気順で絞り込み、
    取得時点の上位5作品を取得します。

    その後、各ポスター画像をK-means法で分析し、
    主要5色・色の割合・明度・彩度を算出します。
    """
)

if st.session_state.retrieved_at:

    st.info(
        "最終取得日時："
        + st.session_state.retrieved_at
    )

else:

    st.warning(
        "左側の「🔄 最新データ取得」を押してください。"
    )


# ============================================================
# データがない場合
# ============================================================

if not movies:
    st.stop()


# ============================================================
# KPI
# ============================================================

valid_movies = [
    movie
    for movie in movies
    if movie["analysis_status"]
    == "OK"
]

col1, col2, col3, col4 = (
    st.columns(4)
)

with col1:

    st.metric(
        "取得作品数",
        len(movies),
    )

with col2:

    st.metric(
        "ジャンル数",
        len(GENRES),
    )

with col3:

    st.metric(
        "1ジャンルあたり",
        "Top 5",
    )

with col4:

    st.metric(
        "カラー分析済み",
        len(valid_movies),
    )


# ============================================================
# ジャンル別作品表示
# ============================================================

st.header(
    "1. 取得時点の各ジャンル Top 5"
)

for genre_name in GENRES:

    st.markdown(
        f"## {genre_name}"
    )

    genre_movies = [
        movie
        for movie in movies
        if movie["genre"]
        == genre_name
    ]

    columns = st.columns(5)

    for index, movie in enumerate(
        genre_movies[:5]
    ):

        with columns[index]:

            st.markdown(
                '<div class="movie-card">',
                unsafe_allow_html=True,
            )

            st.image(
                movie["poster_url"],
                use_container_width=True,
            )

            st.markdown(
                f"### {movie['title']}"
            )

            st.write(
                f"公開年：{movie['year']}"
            )

            st.write(
                f"評価：⭐ {movie['rating']:.1f}"
            )

            st.write(
                f"投票数：{movie['vote_count']:,}"
            )

            st.caption(
                movie["overview"]
            )

            if movie["palette"]:

                st.markdown(
                    "**K-means 主要5色**"
                )

                for color in (
                    movie["palette"]
                ):

                    hex_color = (
                        color["hex"]
                    )

                    percentage = (
                        color[
                            "percentage"
                        ]
                    )

                    st.markdown(
                        f"""
                        <div
                            class="color-bar"
                            style="
                                background-color:{hex_color};
                                width:{max(percentage, 3)}%;
                            "
                        ></div>

                        <div class="small">
                            {hex_color} /
                            {percentage}%
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            else:

                st.warning(
                    "カラー分析できませんでした。"
                )

            st.markdown(
                "</div>",
                unsafe_allow_html=True,
            )


# ============================================================
# Plotly用DataFrame
# ============================================================

rows = []

for movie in valid_movies:

    rows.append(
        {
            "genre": movie["genre"],
            "title": movie["title"],
            "year": movie["year"],
            "rating": movie["rating"],
            "brightness": movie[
                "brightness"
            ],
            "saturation": movie[
                "saturation"
            ],
        }
    )

df = pd.DataFrame(rows)


# ============================================================
# 色彩散布図
# ============================================================

st.header(
    "2. ジャンル別 色彩トーン比較"
)

st.markdown(
    """
    **横軸：彩度（鮮やかさ）**

    **縦軸：明度（明るさ）**

    右上ほど「明るく鮮やか」、
    左下ほど「暗く落ち着いた」色彩です。
    """
)

if not df.empty:

    fig = px.scatter(
        df,
        x="saturation",
        y="brightness",
        color="genre",
        hover_name="title",
        hover_data={
            "year": True,
            "rating": True,
            "saturation": ":.1f",
            "brightness": ":.1f",
        },
        labels={
            "saturation": "彩度（鮮やかさ）",
            "brightness": "明度（明るさ）",
            "genre": "ジャンル",
            "year": "公開年",
            "rating": "評価",
        },
        title=(
            "取得時点のTop 5・明度 × 彩度"
        ),
    )

    fig.update_xaxes(
        range=[0, 100]
    )

    fig.update_yaxes(
        range=[0, 100]
    )

    fig.update_layout(
        height=600
    )

    st.plotly_chart(
        fig,
        use_container_width=True,
    )

    summary = (
        df.groupby("genre")
        .agg(
            平均明度=(
                "brightness",
                "mean",
            ),
            平均彩度=(
                "saturation",
                "mean",
            ),
            平均評価=(
                "rating",
                "mean",
            ),
        )
        .round(1)
        .reset_index()
    )

    st.subheader(
        "ジャンル別平均"
    )

    st.dataframe(
        summary,
        use_container_width=True,
        hide_index=True,
    )


# ============================================================
# 取得データ一覧
# ============================================================

st.header(
    "3. 取得した25作品の一覧"
)

table_rows = []

for movie in movies:

    table_rows.append(
        {
            "ジャンル": movie[
                "genre"
            ],
            "タイトル": movie[
                "title"
            ],
            "公開年": movie[
                "year"
            ],
            "評価": movie[
                "rating"
            ],
            "投票数": movie[
                "vote_count"
            ],
            "明度": movie[
                "brightness"
            ],
            "彩度": movie[
                "saturation"
            ],
            "TMDb ID": movie[
                "tmdb_id"
            ],
        }
    )

table_df = pd.DataFrame(
    table_rows
)

st.dataframe(
    table_df,
    use_container_width=True,
    hide_index=True,
)


# ============================================================
# TMDb注意書き
# ============================================================

st.markdown("---")

st.caption(
    "This product uses the TMDB API "
    "but is not endorsed or certified by TMDB."
)

st.caption(
    "映画情報・ポスター画像はTMDbから取得しています。"
    "利用時はTMDbの利用規約・APIポリシーを確認してください。"
)

st.caption(
    "※「Top 5」はTMDb Discover APIの "
    "popularity.desc を基準にしています。"
)