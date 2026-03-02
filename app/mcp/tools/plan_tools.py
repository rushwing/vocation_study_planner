"""Plan MCP tools: targets and AI plan generation (role: best_pal/admin)."""

import logging
from datetime import date
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import selectinload

from app.database import AsyncSessionLocal
from app.mcp.auth import Role, require_role, verify_best_pal_owns_go_getter
from app.mcp.server import mcp
from app.crud import crud_go_getter, crud_target, crud_plan
from app.schemas.target import TargetCreate, TargetUpdate
from app.schemas.plan import PlanUpdate
from app.models.plan import Plan
from app.models.weekly_milestone import WeeklyMilestone
from app.models.task import Task
from app.models.target import VacationType, TargetStatus
from app.services import plan_generator, github_service

logger = logging.getLogger(__name__)


def _require_chat_id(chat_id: Optional[int]) -> int:
    if chat_id is None:
        raise ValueError("X-Telegram-Chat-Id header is required")
    return chat_id


@mcp.tool()
async def create_target(
    go_getter_id: int,
    title: str,
    subject: str,
    description: str,
    subcategory_id: Optional[int] = None,
    vacation_type: str = "summer",
    vacation_year: int = 2026,
    priority: int = 3,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Create a learning target for a go getter. Requires best_pal/admin role.

    subcategory_id: use list_track_categories to discover valid IDs.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        schema = TargetCreate(
            go_getter_id=go_getter_id,
            title=title,
            subject=subject,
            description=description,
            subcategory_id=subcategory_id,
            vacation_type=VacationType(vacation_type),
            vacation_year=vacation_year,
            priority=priority,
        )
        target = await crud_target.create(db, obj_in=schema)
        await db.commit()
        return {
            "id": target.id,
            "title": target.title,
            "subject": target.subject,
            "go_getter_id": target.go_getter_id,
        }


@mcp.tool()
async def update_target(
    target_id: int,
    title: Optional[str] = None,
    subject: Optional[str] = None,
    description: Optional[str] = None,
    priority: Optional[int] = None,
    status: Optional[str] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Update a learning target. Requires best_pal/admin role."""
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        target = await crud_target.get(db, target_id)
        if not target:
            raise ValueError(f"Target {target_id} not found")
        await verify_best_pal_owns_go_getter(db, caller_id, target.go_getter_id)
        schema = TargetUpdate(
            title=title,
            subject=subject,
            description=description,
            priority=priority,
            status=TargetStatus(status) if status else None,
        )
        target = await crud_target.update(db, db_obj=target, obj_in=schema)
        await db.commit()
        return {"id": target.id, "title": target.title, "status": target.status.value}


@mcp.tool()
async def delete_target(
    target_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Cancel a learning target by setting its status to 'cancelled' (safe soft-delete).

    This preserves all associated plans, milestones, and check-in history.
    Physical deletion is intentionally blocked when a target has plans.
    Requires admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin])
        target = await crud_target.get(db, target_id)
        if not target:
            raise ValueError(f"Target {target_id} not found")
        target.status = TargetStatus.cancelled
        db.add(target)
        await db.commit()
        return {"success": True, "target_id": target_id, "status": "cancelled"}


@mcp.tool()
async def list_targets(
    go_getter_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> list[dict]:
    """List targets for a go getter. Requires best_pal/admin role."""
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        await verify_best_pal_owns_go_getter(db, caller_id, go_getter_id)
        targets = await crud_target.get_by_go_getter(db, go_getter_id)
        return [
            {"id": t.id, "title": t.title, "subject": t.subject, "status": t.status.value}
            for t in targets
        ]


@mcp.tool()
async def generate_plan(
    target_id: int,
    start_date: str,
    end_date: str,
    daily_study_minutes: int = 60,
    preferred_days: Optional[list[int]] = None,
    extra_instructions: Optional[str] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """
    [ADVANCED] Generate a standalone AI study plan directly for a single target.

    IMPORTANT: For new GoalGroup creation, always use the wizard flow instead:
      start_goal_group_wizard → set_wizard_scope → set_wizard_targets
        → set_wizard_constraints → confirm_goal_group

    Use generate_plan ONLY when you need to regenerate a plan for a single
    target outside the wizard (e.g. admin re-planning, one-off plan fix).
    Automatically commits the plan to GitHub. Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    if preferred_days is None:
        preferred_days = [0, 1, 2, 3, 4]  # Mon-Fri default

    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)

    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])

        target = await crud_target.get(db, target_id)
        if not target:
            raise ValueError(f"Target {target_id} not found")

        await verify_best_pal_owns_go_getter(db, caller_id, target.go_getter_id)

        go_getter = await crud_go_getter.get(db, target.go_getter_id)
        if not go_getter:
            raise ValueError(f"Go getter {target.go_getter_id} not found")

        plan = await plan_generator.generate_plan(
            db=db,
            target=target,
            pupil_name=go_getter.name,
            grade=go_getter.grade,
            start_date=start,
            end_date=end,
            daily_study_minutes=daily_study_minutes,
            preferred_days=preferred_days,
            extra_instructions=extra_instructions,
        )

        # Build Markdown for GitHub
        result = await db.execute(
            select(Plan)
            .options(selectinload(Plan.milestones).selectinload(WeeklyMilestone.tasks))
            .where(Plan.id == plan.id)
        )
        full_plan = result.scalar_one()
        md = _plan_to_markdown(full_plan, go_getter.name, target)

        try:
            sha, path = await github_service.commit_plan(
                go_getter.name, target.vacation_type.value, target.vacation_year, plan.title, md
            )
            plan.github_commit_sha = sha
            plan.github_file_path = path
        except Exception as exc:
            logger.warning("GitHub commit failed: %s", exc)

        await db.commit()

        return {
            "plan_id": plan.id,
            "title": plan.title,
            "start_date": str(plan.start_date),
            "end_date": str(plan.end_date),
            "total_weeks": plan.total_weeks,
            "status": plan.status.value,
            "github_file_path": plan.github_file_path,
            "milestones": len(full_plan.milestones),
        }


def _plan_to_markdown(plan, pupil_name: str, target) -> str:
    day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    lines = [
        f"# {plan.title}",
        f"**Go Getter:** {pupil_name}",
        f"**Subject:** {target.subject}",
        f"**Period:** {plan.start_date} – {plan.end_date}",
        f"",
        f"## Overview",
        f"{plan.overview}",
        f"",
    ]
    for ms in plan.milestones:
        lines += [
            f"## Week {ms.week_number}: {ms.title}",
            f"*{ms.start_date} – {ms.end_date}*",
            f"",
            f"{ms.description}",
            f"",
            "| Day | Task | Type | Minutes | XP |",
            "|-----|------|------|---------|-----|",
        ]
        for task in ms.tasks:
            day = day_names[task.day_of_week]
            opt = " *(opt)*" if task.is_optional else ""
            lines.append(
                f"| {day} | {task.title}{opt} | {task.task_type.value} | "
                f"{task.estimated_minutes} | {task.xp_reward} |"
            )
        lines.append("")
    return "\n".join(lines)


@mcp.tool()
async def update_plan(
    plan_id: int,
    title: Optional[str] = None,
    status: Optional[str] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Update a plan's title or status. Requires best_pal/admin role."""
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        plan = await crud_plan.get(db, plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        target = await crud_target.get(db, plan.target_id)
        await verify_best_pal_owns_go_getter(db, caller_id, target.go_getter_id)
        from app.models.plan import PlanStatus

        schema = PlanUpdate(
            title=title,
            status=PlanStatus(status) if status else None,
        )
        plan = await crud_plan.update(db, db_obj=plan, obj_in=schema)
        await db.commit()
        return {"id": plan.id, "title": plan.title, "status": plan.status.value}


@mcp.tool()
async def cancel_plan(
    plan_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Cancel a plan by setting its status to 'cancelled' (safe soft-delete).

    This is the correct way to discard or replace a plan. It preserves all
    milestones, tasks and check-in history for audit purposes. Physical
    deletion is intentionally not supported through the conversation interface.

    Typical "delete and rebuild" flow:
      1. cancel_plan(plan_id)          ← marks old plan as cancelled
      2. start_goal_group_wizard(...)  ← guided Q&A to build the new plan
      3. set_wizard_scope / targets / constraints / confirm_goal_group

    Requires best_pal/admin role.
    """
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        plan = await crud_plan.get(db, plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        target = await crud_target.get(db, plan.target_id)
        await verify_best_pal_owns_go_getter(db, caller_id, target.go_getter_id)
        from app.models.plan import PlanStatus

        plan.status = PlanStatus.cancelled
        db.add(plan)
        await db.commit()
        return {
            "success": True,
            "plan_id": plan_id,
            "status": "cancelled",
            "message": "Plan cancelled. Use the wizard flow to create a replacement.",
        }


@mcp.tool()
async def list_plans(
    go_getter_id: Optional[int] = None,
    target_id: Optional[int] = None,
    x_telegram_chat_id: Optional[int] = None,
) -> list[dict]:
    """List plans. Requires best_pal/admin role."""
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        if go_getter_id:
            plans = await crud_plan.get_by_go_getter(db, go_getter_id, target_id)
        else:
            plans = await crud_plan.get_multi(db)
        return [
            {
                "id": p.id,
                "title": p.title,
                "status": p.status.value,
                "start_date": str(p.start_date),
                "end_date": str(p.end_date),
            }
            for p in plans
        ]


@mcp.tool()
async def get_plan_detail(
    plan_id: int,
    x_telegram_chat_id: Optional[int] = None,
) -> dict:
    """Get full plan with milestones and tasks. Requires best_pal/admin role."""
    caller_id = _require_chat_id(x_telegram_chat_id)
    async with AsyncSessionLocal() as db:
        await require_role(db, caller_id, [Role.admin, Role.best_pal])
        plan = await crud_plan.get_with_milestones(db, plan_id)
        if not plan:
            raise ValueError(f"Plan {plan_id} not found")
        target = await crud_target.get(db, plan.target_id)
        await verify_best_pal_owns_go_getter(db, caller_id, target.go_getter_id)
        return {
            "id": plan.id,
            "title": plan.title,
            "overview": plan.overview,
            "status": plan.status.value,
            "start_date": str(plan.start_date),
            "end_date": str(plan.end_date),
            "total_weeks": plan.total_weeks,
            "milestones": [
                {
                    "week_number": ms.week_number,
                    "title": ms.title,
                    "total_tasks": ms.total_tasks,
                    "completed_tasks": ms.completed_tasks,
                    "tasks": [
                        {
                            "id": t.id,
                            "title": t.title,
                            "day_of_week": t.day_of_week,
                            "estimated_minutes": t.estimated_minutes,
                            "xp_reward": t.xp_reward,
                        }
                        for t in ms.tasks
                    ],
                }
                for ms in plan.milestones
            ],
        }
