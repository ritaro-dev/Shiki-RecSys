from fastapi import APIRouter, status

router = APIRouter(
    prefix="/health",
    tags=["health"],
)


@router.get(
    "",
    status_code=status.HTTP_200_OK,
)
def get_health() -> dict[str, str]:
    """
    Проверяет, что HTTP-приложение запущено
    и принимает запросы.
    """

    return {
        "status": "ok",
    }
