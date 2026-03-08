from datetime import date
from typing import TYPE_CHECKING, Optional

from sqlalchemy import Date, Enum, ForeignKey, Integer, SmallInteger, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

if TYPE_CHECKING:
    from app.models.goal_group import GoalGroup
    from app.models.target import Target
    from app.models.weekly_milestone import WeeklyMilestone

import enum


class PlanStatus(str, enum.Enum):
    draft = "draft"
    active = "active"
    completed = "completed"
    cancelled = "cancelled"


class Plan(Base, TimestampMixin):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    target_id: Mapped[int] = mapped_column(Integer, ForeignKey("targets.id"), nullable=False)
    title: Mapped[str] = mapped_column(String(200), nullable=False)
    overview: Mapped[str] = mapped_column(Text, nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_weeks: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    status: Mapped[PlanStatus] = mapped_column(
        Enum(PlanStatus), nullable=False, default=PlanStatus.draft
    )
    github_commit_sha: Mapped[Optional[str]] = mapped_column(String(40), nullable=True)
    github_file_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    llm_prompt_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    llm_completion_tokens: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    # Re-planning versioning: incremented each time a plan is superseded and regenerated
    version: Mapped[int] = mapped_column(SmallInteger, nullable=False, default=1)
    # Points to the newer Plan that replaced this one (null if this is the current plan)
    superseded_by_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("plans.id"), nullable=True
    )
    # GoalGroup context this plan was generated within (nullable for standalone plans)
    group_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("goal_groups.id"), nullable=True
    )
    # Wizard that generated this plan as a draft (null for standalone / re-plan flows)
    wizard_id: Mapped[Optional[int]] = mapped_column(
        Integer, ForeignKey("goal_group_wizards.id", ondelete="SET NULL"), nullable=True
    )

    # Relationships
    target: Mapped["Target"] = relationship("Target", back_populates="plans")
    milestones: Mapped[list["WeeklyMilestone"]] = relationship(
        "WeeklyMilestone", back_populates="plan", order_by="WeeklyMilestone.week_number"
    )
    superseded_by: Mapped[Optional["Plan"]] = relationship(
        "Plan", foreign_keys=[superseded_by_id], remote_side="Plan.id"
    )
    group: Mapped[Optional["GoalGroup"]] = relationship("GoalGroup")
