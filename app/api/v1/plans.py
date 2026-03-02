"""Plan and target endpoints."""

from datetime import date
from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select
from sqlalchemy.orm import selectinload
from sqlalchemy.ext.asyncio import AsyncSession
from app.database import get_db
from app.api.v1.deps import require_best_pal_or_admin, require_admin, verify_best_pal_owns_go_getter
from app.crud import crud_target, crud_plan, crud_go_getter
from app.services.goal_group_service import assert_subcategory_available
from app.schemas.target import TargetCreate, TargetUpdate, TargetResponse
from app.schemas.plan import PlanUpdate, PlanResponse, GeneratePlanRequest
from app.services import plan_generator, github_service
from app.models.plan import Plan
from app.models.weekly_milestone import WeeklyMilestone
from app.models.target import VacationType

router = APIRouter(tags=["plans"])


@router.get("/targets", response_model=list[TargetResponse])
async def list_targets(
    go_getter_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    await verify_best_pal_owns_go_getter(go_getter_id, chat_id, db)
    return await crud_target.get_by_go_getter(db, go_getter_id)


@router.post("/targets", response_model=TargetResponse, status_code=201)
async def create_target(
    body: TargetCreate,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    await verify_best_pal_owns_go_getter(body.go_getter_id, chat_id, db)
    if body.subcategory_id is not None:
        try:
            await assert_subcategory_available(
                db, go_getter_id=body.go_getter_id, subcategory_id=body.subcategory_id
            )
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
    return await crud_target.create(db, obj_in=body)


@router.patch("/targets/{target_id}", response_model=TargetResponse)
async def update_target(
    target_id: int,
    body: TargetUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[int, Depends(require_best_pal_or_admin)],
):
    target = await crud_target.get(db, target_id)
    if not target:
        raise HTTPException(404, "Target not found")
    # If subcategory is being changed, re-check uniqueness (exclude self)
    new_sub = body.subcategory_id
    if new_sub is not None and new_sub != target.subcategory_id:
        try:
            await assert_subcategory_available(
                db,
                go_getter_id=target.go_getter_id,
                subcategory_id=new_sub,
                exclude_target_id=target_id,
            )
        except ValueError as e:
            raise HTTPException(409, str(e)) from e
    return await crud_target.update(db, db_obj=target, obj_in=body)


@router.delete("/targets/{target_id}")
async def delete_target(
    target_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[int, Depends(require_admin)],
):
    """Physical delete — admin only. Blocked if the target already has plans.

    Targets with plans cannot be physically deleted to preserve audit history.
    Use PATCH /targets/{target_id} with {"status": "cancelled"} to deactivate instead.
    """
    from sqlalchemy import func
    from app.models.plan import Plan

    t = await crud_target.get(db, target_id)
    if not t:
        raise HTTPException(404, "Target not found")
    result = await db.execute(
        select(func.count()).select_from(Plan).where(Plan.target_id == target_id)
    )
    plan_count = result.scalar_one()
    if plan_count > 0:
        raise HTTPException(
            409,
            f"Target {target_id} has {plan_count} plan(s) and cannot be physically deleted. "
            "Set status='cancelled' via PATCH /targets/{target_id} to deactivate it instead.",
        )
    await crud_target.remove(db, id=target_id)
    return {"success": True}


@router.get("/plans", response_model=list[PlanResponse])
async def list_plans(
    go_getter_id: Optional[int] = None,
    target_id: Optional[int] = None,
    db: AsyncSession = Depends(get_db),
    chat_id: int = Depends(require_best_pal_or_admin),
):
    if go_getter_id:
        await verify_best_pal_owns_go_getter(go_getter_id, chat_id, db)
        return await crud_plan.get_by_go_getter(db, go_getter_id, target_id)
    return await crud_plan.get_multi(db)


@router.post("/plans/generate", status_code=201)
async def generate_plan(
    body: GeneratePlanRequest,
    db: Annotated[AsyncSession, Depends(get_db)],
    chat_id: Annotated[int, Depends(require_best_pal_or_admin)],
):
    target = await crud_target.get(db, body.target_id)
    if not target:
        raise HTTPException(404, "Target not found")
    go_getter = await crud_go_getter.get(db, target.go_getter_id)
    if not go_getter:
        raise HTTPException(404, "Go getter not found")
    await verify_best_pal_owns_go_getter(target.go_getter_id, chat_id, db)

    plan = await plan_generator.generate_plan(
        db=db,
        target=target,
        pupil_name=go_getter.name,
        grade=go_getter.grade,
        start_date=body.start_date,
        end_date=body.end_date,
        daily_study_minutes=body.daily_study_minutes,
        preferred_days=body.preferred_days,
        extra_instructions=body.extra_instructions,
    )

    from app.mcp.tools.plan_tools import _plan_to_markdown

    full_plan = (
        await db.execute(
            select(Plan)
            .options(selectinload(Plan.milestones).selectinload(WeeklyMilestone.tasks))
            .where(Plan.id == plan.id)
        )
    ).scalar_one()

    md = _plan_to_markdown(full_plan, go_getter.name, target)
    try:
        sha, path = await github_service.commit_plan(
            go_getter.name, target.vacation_type.value, target.vacation_year, plan.title, md
        )
        plan.github_commit_sha = sha
        plan.github_file_path = path
        await db.flush()
    except Exception:
        pass

    return {
        "plan_id": plan.id,
        "title": plan.title,
        "start_date": str(plan.start_date),
        "end_date": str(plan.end_date),
        "total_weeks": plan.total_weeks,
        "status": plan.status.value,
    }


@router.patch("/plans/{plan_id}", response_model=PlanResponse)
async def update_plan(
    plan_id: int,
    body: PlanUpdate,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[int, Depends(require_best_pal_or_admin)],
):
    plan = await crud_plan.get(db, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    return await crud_plan.update(db, db_obj=plan, obj_in=body)


@router.delete("/plans/{plan_id}")
async def delete_plan(
    plan_id: int,
    db: Annotated[AsyncSession, Depends(get_db)],
    _: Annotated[int, Depends(require_admin)],
):
    """Physical delete — admin only. Blocked if the plan already has milestones.

    Plans with milestones/tasks cannot be physically deleted to preserve audit
    history. Use PATCH /plans/{id} with {"status": "cancelled"} instead.
    """
    plan = await crud_plan.get(db, plan_id)
    if not plan:
        raise HTTPException(404, "Plan not found")
    from sqlalchemy import func

    result = await db.execute(
        select(func.count()).select_from(WeeklyMilestone).where(WeeklyMilestone.plan_id == plan_id)
    )
    milestone_count = result.scalar_one()
    if milestone_count > 0:
        raise HTTPException(
            409,
            f"Plan {plan_id} has {milestone_count} milestone(s) and cannot be physically deleted. "
            "Set status='cancelled' via PATCH /plans/{plan_id} to deactivate it instead, "
            "then create a new plan via the wizard flow.",
        )
    await db.delete(plan)
    await db.flush()
    return {"success": True}
