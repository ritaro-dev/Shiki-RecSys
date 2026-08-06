from sqlalchemy import text
from sqlalchemy.orm import Session


class LegacySeedUserRepository:
    """
    Выбирает пользователей из устаревшей
    таблицы user_rates.
    """

    def get_stratified_user_ids(
        self,
        session: Session,
        *,
        total_users: int,
    ) -> list[int]:
        """
        Возвращает стратифицированную выборку
        пользователей по количеству явных оценок.
        """

        if total_users <= 0:
            raise ValueError("total_users должен быть больше 0.")

        quota_10_19 = int(total_users * 0.30)
        quota_20_49 = int(total_users * 0.50)
        quota_50_99 = int(total_users * 0.15)

        quota_100_plus = total_users - quota_10_19 - quota_20_49 - quota_50_99

        statement = text(
            """
            WITH user_activity AS (
                SELECT
                    user_id,
                    COUNT(*) AS ratings_count
                FROM user_rates
                WHERE rating > 0
                GROUP BY user_id
            ),
            selected_users AS (
                (
                    SELECT
                        user_id,
                        1 AS group_order
                    FROM user_activity
                    WHERE ratings_count BETWEEN 10 AND 19
                    ORDER BY
                        MD5(user_id::TEXT),
                        user_id
                    LIMIT :quota_10_19
                )

                UNION ALL

                (
                    SELECT
                        user_id,
                        2 AS group_order
                    FROM user_activity
                    WHERE ratings_count BETWEEN 20 AND 49
                    ORDER BY
                        MD5(user_id::TEXT),
                        user_id
                    LIMIT :quota_20_49
                )

                UNION ALL

                (
                    SELECT
                        user_id,
                        3 AS group_order
                    FROM user_activity
                    WHERE ratings_count BETWEEN 50 AND 99
                    ORDER BY
                        MD5(user_id::TEXT),
                        user_id
                    LIMIT :quota_50_99
                )

                UNION ALL

                (
                    SELECT
                        user_id,
                        4 AS group_order
                    FROM user_activity
                    WHERE ratings_count >= 100
                    ORDER BY
                        MD5(user_id::TEXT),
                        user_id
                    LIMIT :quota_100_plus
                )
            )
            SELECT user_id
            FROM selected_users
            ORDER BY
                group_order,
                MD5(user_id::TEXT),
                user_id;
            """
        )

        result = session.execute(
            statement,
            {
                "quota_10_19": quota_10_19,
                "quota_20_49": quota_20_49,
                "quota_50_99": quota_50_99,
                "quota_100_plus": quota_100_plus,
            },
        )

        return [int(user_id) for user_id in result.scalars().all()]
