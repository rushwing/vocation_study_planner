"""Add wizard_id to plans for draft-plan ownership tracking

Revision ID: 008
Revises: 007
Create Date: 2026-03-09 00:00:00.000000

Changes:
  plans:
    + wizard_id  Integer nullable FK → goal_group_wizards.id
                 SET NULL on wizard delete, so plan cleanup data is not lost.
                 Non-null only for plans generated inside a wizard draft flow.
"""

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "008"
down_revision: Union[str, None] = "007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "plans",
        sa.Column(
            "wizard_id",
            sa.Integer(),
            sa.ForeignKey("goal_group_wizards.id", ondelete="SET NULL"),
            nullable=True,
            comment="Wizard that generated this draft plan; null for standalone plans",
        ),
    )
    op.create_index("ix_plans_wizard_id", "plans", ["wizard_id"])


def downgrade() -> None:
    op.drop_index("ix_plans_wizard_id", table_name="plans")
    op.drop_column("plans", "wizard_id")
