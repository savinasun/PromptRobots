# Viser view

Start a simulation with `--viser --language-output` (as in `scripts/run_example.py`).
The sidebar's **Live notes** section displays the action note as it arrives, before
the gateway validates and executes the motion. It also shows the action state,
optional reasoning summary, and the last 12 activity entries. Streamed previews
never execute partial tool arguments. Action-only mode still shows numeric actions
and any returned reasoning summary, but does not request language notes.

The view starts zoomed onto the table and both arms. **Display → View / Reset view**
can restore the overview or look through a robot camera. Camera frustums and tool
axes can be enabled there; they start hidden. The grid and other helper overlays
are excluded from camera images sent to the model.

## Camera timeouts

3D camera captures require a responsive browser tab with WebGL. A connected tab
can still fail to render if it is suspended, minimized, or its renderer stalls.
The Cameras section shows whether observations use 3D images or schematic fallback.

After a render failure, the runner tries one other connected browser, if available.
Failed clients are skipped for 30 seconds so each observation does not repeatedly
block on the same tab. Only a complete set of fresh frames is used. Open or reload
Viser in a visible tab, then press **Retry 3D cameras** to clear the cooldown for the
next observation. If WebGL is unavailable, enable browser hardware acceleration.

The timeout per frame is configurable with `--set viz.render_timeout_s=15` for a
slow renderer; the retry cooldown uses `--set viz.render_retry_s=30`. Increasing
the timeout cannot make a suspended browser render.

The live notes implementation uses the Responses API's
[streaming events](https://developers.openai.com/api/docs/guides/streaming-responses).
Final responses, usage, and tool validation retain the same logging and gateway path.
