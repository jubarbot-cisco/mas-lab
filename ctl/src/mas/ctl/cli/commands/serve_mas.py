#  Copyright (c) 2026 Cisco Systems, Inc. and its affiliates
#  SPDX-License-Identifier: Apache-2.0
"""mas-ctl serve-mas — hold a MAS open and expose its agents over HTTP.

Same composition flags as ``run-mas``; the difference is who drives the agents.
``run-mas`` drives them from stdin, ``serve-mas`` from REST and A2A, so a remote
client can address each agent individually. Not to be confused with
``mas-lab serve``, which starts the lab controller daemon and runs a MAS as
short-lived subprocess jobs.
"""

from __future__ import annotations

import os

import click
import yaml
from mas.ctl.cli.obs_flags import observability_options, resolve_observability_config
from mas.ctl.cli.runtime_flags import runtime_id_choice
from mas.ctl.cli.trace_flags import trace_options
from mas.ctl.deployment.runtime_id import DEFAULT_RUNTIME_ID


@click.command("serve-mas")
@click.argument("manifest", required=False, type=click.Path())
@click.option("-o", "--overlay", "overlays", multiple=True, type=click.Path())
@click.option("-d", "--deployment", "deployment", default=None, type=click.Path())
@click.option(
    "--flavour",
    default="local",
    show_default=True,
    help="Deployment flavour from library-standard (only 'local' supported for now)",
)
@click.option("--infra-ref", "infra_refs", multiple=True)
@click.option("--kernel", default=DEFAULT_RUNTIME_ID, type=runtime_id_choice())
@click.option("-q", "--query", "queries", multiple=True, help="Single or multi-turn query")
@click.option("--host", default="127.0.0.1", show_default=True, help="Bind address.")
@click.option("--port", default=8080, show_default=True, type=int)
@click.option("--token", default=None, help="Bearer token; else $MAS_SERVE_TOKEN.")
@click.option("--no-auth", is_flag=True, help="Serve without a bearer token.")
@click.option("--no-validate", is_flag=True)
@observability_options
@trace_options
@click.pass_context
def serve_mas_cmd(
    ctx: click.Context,
    manifest: str | None,
    overlays: tuple[str, ...],
    deployment: str | None,
    flavour: str,
    infra_refs: tuple[str, ...],
    kernel: str,
    queries: tuple[str, ...],
    host: str,
    port: int,
    token: str | None,
    no_auth: bool,
    no_validate: bool,
    events,
    events_file,
    events_stdout,
    events_format,
    trace: bool,
    trace_timestamps: bool,
    trace_engine: bool,
    trace_summary: bool,
    trace_color: bool,
) -> None:
    """Serve a MAS over HTTP: address each agent by id via REST or A2A."""
    from mas.ctl.paths import manifest_cwd, resolve_overlay_path

    try:
        import uvicorn
    except ModuleNotFoundError:
        click.echo("error: serve-mas needs extras — pip install 'mas-ctl[serve]'", err=True)
        raise SystemExit(2) from None

    from mas.ctl.executor.run_mas import build_running_mas
    from mas.ctl.serve.app import build_app
    from mas.ctl.serve.cards import build_cards
    from mas.ctl.session.flavour import FlavourError, resolve_flavour

    token = token or os.environ.get("MAS_SERVE_TOKEN")
    if no_auth:
        token = None
    elif not token:
        click.echo(
            "error: a bearer token is required — pass --token, set MAS_SERVE_TOKEN, "
            "or accept the risk with --no-auth",
            err=True,
        )
        raise SystemExit(2)

    if manifest is None:
        manifest = "mas.yaml"
    verbose = int(ctx.obj.get("verbose", 0) if ctx.obj else 0)

    try:
        flavour_spec = resolve_flavour(flavour)
    except FlavourError as exc:
        click.echo(f"error: {exc}", err=True)
        raise SystemExit(2) from None

    with manifest_cwd(manifest, overlay_paths=overlays) as session:
        deployment_path = None
        deployment_doc = None
        if deployment:
            deployment_path = resolve_overlay_path(
                deployment,
                orig_cwd=session.original_cwd,
                manifest_dir=session.manifest_dir,
            )
            deployment_doc = yaml.safe_load(deployment_path.read_text(encoding="utf-8"))

        mas_doc = yaml.safe_load(session.local_manifest.read_text(encoding="utf-8"))
        obs_cfg = resolve_observability_config(
            events=events,
            events_file=events_file,
            events_stdout=events_stdout,
            events_format=events_format,
            manifest=mas_doc,
            deployment=deployment_doc,
            flavour_spec=flavour_spec,
        )

        try:
            mas = build_running_mas(
                session.local_manifest,
                overlay_paths=list(session.overlays),
                overlay_ids=None,
                infra_refs=list(infra_refs),
                deployment_path=deployment_path,
                kernel_backend=kernel,
                validate=not no_validate,
                verbose=verbose,
                manifest_dir=session.manifest_dir,
                obs_config=obs_cfg,
                trace=trace,
                trace_timestamps=trace_timestamps,
                trace_engine=trace_engine or verbose >= 2,
                trace_summary=trace_summary,
                trace_color=trace_color,
            )
        except KeyError as exc:
            click.echo(f"error: {exc}", err=True)
            raise SystemExit(1) from None

        base_url = f"http://{host}:{port}"
        if queries:
            from mas.ctl.session.mas_router import MasAddressRouter

            router = MasAddressRouter(mas)
            for query in queries:
                router.run_turn(query)

        app = build_app(
            mas,
            build_cards(mas.materialized, base_url=base_url, secured=token is not None),
            token=token,
        )
        click.echo(f"serving {len(mas.agent_ids)} agents on {base_url}/agents", err=True)
        try:
            uvicorn.run(app, host=host, port=port, log_level="info")
        finally:
            mas.close()
