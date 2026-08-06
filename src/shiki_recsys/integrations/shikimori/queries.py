USER_HISTORY_QUERY: str = """
query(
    $userId: ID!,
    $page: PositiveInt!,
    $limit: PositiveInt!
) {
    userRates(
        userId: $userId,
        page: $page,
        limit: $limit,
        targetType: Anime
    ) {
        score
        status
        updatedAt

        anime {
            id
        }
    }
}
"""


ANIMES_CATALOG_QUERY: str = """
query(
    $page: PositiveInt!,
    $limit: PositiveInt!
) {
    animes(
        page: $page,
        limit: $limit,
        order: popularity
    ) {
        id
        name
        russian
        score
        kind
        status
        episodes
        duration
        rating

        genres {
            name
        }

        studios {
            name
        }

        statusesStats {
            status
            count
        }
    }
}
"""
