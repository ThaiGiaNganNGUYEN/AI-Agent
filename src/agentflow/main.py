"""Process entrypoint. Importing this module builds the default app."""

from agentflow.api.app import create_app

app = create_app()


def main() -> None:
    import uvicorn

    uvicorn.run("agentflow.main:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
