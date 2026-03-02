"""LLM-powered vacation study plan generator."""

import json
import logging
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.models.plan import Plan, PlanStatus
from app.models.target import Target
from app.models.weekly_milestone import WeeklyMilestone
from app.models.task import Task, TaskType
from app.services import llm_service

logger = logging.getLogger(__name__)

PLAN_SYSTEM_PROMPT = """\
You are an expert educational curriculum designer specialising in personalised vacation study plans \
for school-age children. Generate a structured, age-appropriate study plan in JSON format.

The JSON must match this exact schema:
{
  "title": "string – short plan title",
  "overview": "string – 2-3 paragraph summary of the plan",
  "weeks": [
    {
      "week_number": 1,
      "title": "string",
      "description": "string – week focus",
      "tasks": [
        {
          "day_of_week": 0,
          "sequence_in_day": 1,
          "title": "string",
          "description": "string – specific instructions for the student",
          "estimated_minutes": 30,
          "task_type": "reading|writing|math|practice|review|project|quiz|other",
          "xp_reward": 10,
          "is_optional": false
        }
      ]
    }
  ]
}

Rules:
- day_of_week: 0=Monday … 6=Sunday
- Only schedule tasks on preferred study days
- estimated_minutes should fit within daily_study_minutes
- xp_reward should be proportional to estimated_minutes (roughly 1 XP per minute, max 60)
- Make descriptions specific and actionable for the student
- Output ONLY the JSON object, no markdown fences
"""


_DEFAULT_DAILY_MINUTES = 60
_DEFAULT_PREFERRED_DAYS = [0, 1, 2, 3, 4, 5, 6]  # all days


def _build_user_prompt(
    target: Target,
    pupil_name: str,
    grade: str,
    start_date: date,
    end_date: date,
    daily_study_minutes: int,
    preferred_days: list[int],
    extra_instructions: str | None,
    reference_materials: list[dict] | None = None,
) -> str:
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    pref_day_names = [day_names[d] for d in sorted(preferred_days)]
    total_days = (end_date - start_date).days + 1
    total_weeks = max(1, (total_days + 6) // 7)

    prompt = (
        f"Go Getter: {pupil_name} (Grade {grade})\n"
        f"Subject: {target.subject}\n"
        f"Learning goal: {target.title}\n"
        f"Description: {target.description}\n"
        f"Study period: {start_date} to {end_date} ({total_weeks} week(s))\n"
        f"Daily study time available: {daily_study_minutes} minutes\n"
        f"Preferred study days: {', '.join(pref_day_names)}\n"
    )
    if extra_instructions:
        prompt += f"Extra instructions: {extra_instructions}\n"
    if reference_materials:
        prompt += "\n## 参考教学资料（来自小红书等平台）\n"
        for mat in reference_materials[:5]:
            prompt += f"- 《{mat['title']}》({mat.get('source', '')})\n"
            for kp in mat.get("key_points", [])[:4]:
                prompt += f"  • {kp}\n"
    return prompt


async def generate_plan(
    db: AsyncSession,
    target: Target,
    pupil_name: str,
    grade: str,
    start_date: date,
    end_date: date,
    daily_study_minutes: int | None = None,
    preferred_days: list[int] | None = None,
    extra_instructions: str | None = None,
    reference_materials: list[dict] | None = None,
    initial_status: PlanStatus = PlanStatus.active,
    deactivate_existing: bool = True,
) -> Plan:
    """
    Call Kimi to generate a structured plan, persist to DB, return Plan object.

    deactivate_existing: when False (wizard draft flow), skip marking existing active
    plans as completed so live plans are not mutated before the wizard is confirmed.
    """
    user_prompt = _build_user_prompt(
        target,
        pupil_name,
        grade,
        start_date,
        end_date,
        daily_study_minutes if daily_study_minutes is not None else _DEFAULT_DAILY_MINUTES,
        preferred_days if preferred_days is not None else _DEFAULT_PREFERRED_DAYS,
        extra_instructions,
        reference_materials=reference_materials,
    )

    plan_data: dict[str, Any] | None = None
    prompt_tokens = completion_tokens = 0

    for attempt in range(3):
        try:
            content, pt, ct = await llm_service.chat_complete_long(
                messages=[
                    {"role": "system", "content": PLAN_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.3,
                max_tokens=8192,
            )
            prompt_tokens += pt
            completion_tokens += ct
            plan_data = json.loads(content)
            break
        except json.JSONDecodeError as exc:
            logger.warning("Plan JSON parse error (attempt %d): %s", attempt + 1, exc)
            if attempt == 2:
                raise ValueError(f"LLM returned invalid JSON after 3 attempts: {exc}") from exc

    assert plan_data is not None

    total_weeks = max(1, (((end_date - start_date).days + 1) + 6) // 7)

    # Deactivate any existing active plan for this specific target.
    # One active plan per target — not per go_getter — to allow parallel tracks
    # (e.g. study + fitness running concurrently).
    # Skipped when deactivate_existing=False (wizard draft flow) so that live plans
    # are not mutated until the wizard is confirmed.
    from sqlalchemy import select as _select

    if deactivate_existing:
        existing_active = await db.execute(
            _select(Plan).where(Plan.target_id == target.id, Plan.status == PlanStatus.active)
        )
        for old_plan in existing_active.scalars().all():
            old_plan.status = PlanStatus.completed
        await db.flush()

    # Create Plan record
    plan = Plan(
        target_id=target.id,
        title=plan_data.get("title", f"{target.subject} Study Plan"),
        overview=plan_data.get("overview", ""),
        start_date=start_date,
        end_date=end_date,
        total_weeks=total_weeks,
        status=initial_status,
        llm_prompt_tokens=prompt_tokens,
        llm_completion_tokens=completion_tokens,
    )
    db.add(plan)
    await db.flush()

    # Create milestones and tasks
    for week_data in plan_data.get("weeks", []):
        week_num = week_data["week_number"]
        week_offset = (week_num - 1) * 7
        ms_start = start_date + timedelta(days=week_offset)
        ms_end = min(ms_start + timedelta(days=6), end_date)

        tasks_data = week_data.get("tasks", [])
        milestone = WeeklyMilestone(
            plan_id=plan.id,
            week_number=week_num,
            title=week_data.get("title", f"Week {week_num}"),
            description=week_data.get("description", ""),
            start_date=ms_start,
            end_date=ms_end,
            total_tasks=len(tasks_data),
            completed_tasks=0,
        )
        db.add(milestone)
        await db.flush()

        for task_data in tasks_data:
            task_type_str = task_data.get("task_type", "practice")
            try:
                task_type = TaskType(task_type_str)
            except ValueError:
                task_type = TaskType.other

            task = Task(
                milestone_id=milestone.id,
                day_of_week=task_data.get("day_of_week", 0),
                sequence_in_day=task_data.get("sequence_in_day", 1),
                title=task_data.get("title", "Study Task"),
                description=task_data.get("description", ""),
                estimated_minutes=task_data.get("estimated_minutes", 30),
                xp_reward=task_data.get("xp_reward", 10),
                task_type=task_type,
                is_optional=task_data.get("is_optional", False),
            )
            db.add(task)

    await db.flush()
    await db.refresh(plan)
    return plan
