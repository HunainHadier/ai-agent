import argparse
import json

from app.builder import AgentBuilder, AgentSpec
from app.runtime import run_agent


def build_command(args: argparse.Namespace) -> None:
    builder = AgentBuilder()
    postman_vars = None
    if args.postman_vars:
        postman_vars = json.loads(args.postman_vars)
    spec = AgentSpec(
        name=args.name,
        description=args.description,
        services=args.services,
        flow=args.flow,
        device_id=args.device_id,
        postman_path=args.postman,
        file_paths=args.files,
        postman_variables=postman_vars,
    )
    path = builder.build(spec)
    print(f"Agent created at {path}")


def run_command(args: argparse.Namespace) -> None:
    output = run_agent(args.device_id, args.name, args.input)
    print(output)


def serve_command(args: argparse.Namespace) -> None:
    import uvicorn

    uvicorn.run("app.main:app", host=args.host, port=args.port, reload=args.reload)


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent builder CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_parser = subparsers.add_parser("build", help="Create a new agent")
    build_parser.add_argument("--name", required=True)
    build_parser.add_argument("--description", required=True)
    build_parser.add_argument("--services", required=True)
    build_parser.add_argument("--flow", required=True)
    build_parser.add_argument("--postman")
    build_parser.add_argument("--files", nargs="*", default=[])
    build_parser.add_argument("--postman-vars")
    build_parser.add_argument("--device-id")
    build_parser.set_defaults(func=build_command)

    run_parser = subparsers.add_parser("run", help="Run an existing agent")
    run_parser.add_argument("--name", required=True)
    run_parser.add_argument("--input", required=True)
    run_parser.add_argument("--device-id")
    run_parser.set_defaults(func=run_command)

    serve_parser = subparsers.add_parser("serve", help="Run API server")
    serve_parser.add_argument("--host", default="0.0.0.0")
    serve_parser.add_argument("--port", type=int, default=8000)
    serve_parser.add_argument("--reload", action="store_true")
    serve_parser.set_defaults(func=serve_command)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
