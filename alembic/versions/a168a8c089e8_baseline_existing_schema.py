"""baseline existing schema

Revision ID: a168a8c089e8
Revises:
Create Date: 2026-08-03 11:52:08.176858
"""

from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "a168a8c089e8"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """
    Создаёт исходную управляемую схему проекта.
    """

    op.create_table(
        "animes",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "name",
            sa.Text(),
            nullable=False,
        ),
        sa.Column(
            "russian_name",
            sa.Text(),
            nullable=True,
        ),
        sa.Column(
            "kind",
            sa.String(length=50),
            nullable=True,
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=True,
        ),
        sa.Column(
            "score",
            sa.Numeric(precision=4, scale=2),
            nullable=True,
        ),
        sa.Column(
            "score_std",
            sa.Numeric(precision=6, scale=4),
            nullable=True,
        ),
        sa.Column(
            "episodes",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "duration",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "rating",
            sa.String(length=20),
            nullable=True,
        ),
        sa.Column(
            "genres",
            postgresql.ARRAY(sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "studios",
            postgresql.ARRAY(sa.Text()),
            nullable=True,
        ),
        sa.Column(
            "stat_completed",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "stat_dropped",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "stat_watching",
            sa.Integer(),
            nullable=True,
        ),
        sa.Column(
            "stat_planned",
            sa.Integer(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("id"),
    )

    op.create_table(
        "svd_collection_progress",
        sa.Column(
            "user_id",
            sa.Integer(),
            autoincrement=False,
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=20),
            nullable=False,
        ),
        sa.Column(
            "rates_saved",
            sa.Integer(),
            server_default=sa.text("0"),
            nullable=False,
        ),
        sa.Column(
            "finished_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.Column(
            "error_message",
            sa.Text(),
            nullable=True,
        ),
        sa.PrimaryKeyConstraint("user_id"),
    )

    op.create_index(
        "idx_svd_collection_progress_status",
        "svd_collection_progress",
        ["status"],
        unique=False,
    )

    op.create_table(
        "user_rates_svd",
        sa.Column(
            "id",
            sa.Integer(),
            autoincrement=True,
            nullable=False,
        ),
        sa.Column(
            "user_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "anime_id",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "rating",
            sa.Integer(),
            nullable=False,
        ),
        sa.Column(
            "status",
            sa.String(length=50),
            nullable=True,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=True,
        ),
        sa.CheckConstraint(
            "rating >= 0 AND rating <= 10",
            name="user_rates_svd_rating_check",
        ),
        sa.ForeignKeyConstraint(
            ["anime_id"],
            ["animes.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "user_id",
            "anime_id",
            name="unique_svd_user_anime",
        ),
    )

    op.create_index(
        "idx_user_rates_svd_user_id",
        "user_rates_svd",
        ["user_id"],
        unique=False,
    )

    op.create_index(
        "idx_user_rates_svd_anime_id",
        "user_rates_svd",
        ["anime_id"],
        unique=False,
    )


def downgrade() -> None:
    """
    Удаляет исходную управляемую схему проекта.
    """

    op.drop_index(
        "idx_user_rates_svd_anime_id",
        table_name="user_rates_svd",
    )

    op.drop_index(
        "idx_user_rates_svd_user_id",
        table_name="user_rates_svd",
    )

    op.drop_table("user_rates_svd")

    op.drop_index(
        "idx_svd_collection_progress_status",
        table_name="svd_collection_progress",
    )

    op.drop_table("svd_collection_progress")
    op.drop_table("animes")
