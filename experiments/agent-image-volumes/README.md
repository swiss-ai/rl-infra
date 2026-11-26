# Agent binaries as image volumes: reference image (chisel)

The RL runs need a few binaries inside every OpenSandbox pod on Breithorn: chisel (tunnel to the model server),
and the agent harnesses (OpenCode, pi, later Claude Code). Today the Gym adapters upload them into each pod.
Proposal by the Breithorn admin: one data-only image per binary in the registry, mounted read-only by the
sandbox pod template as a Kubernetes image volume. This directory holds a reference image to test that.

## The image

`chisel` 1.12.0 (github.com/jpillora/chisel, MIT), linux/amd64, built `FROM scratch`: no OS, no shell, no
entrypoint, one layer. Image root = mount point:

```
/1.12.0/chisel        static binary, 11.6 MB, mode 0555
/1.12.0/LICENSE       upstream MIT licence
/1.12.0/SHA256SUMS    checksums of the two files above
```

Versioned directories: one image can carry several versions later, so pushing an update never changes the
binary that a running training uses (it asks for its own version's path).

## Build (does not push)

```
WORK=<build dir> bash experiments/agent-image-volumes/build.sh
```

Fetches the binary with the pinned release checksum (`backends/nemo-rl/apertus2/fetch-chisel.sh`) and the
licence from the same tag, builds with podman (one squashed layer), mounts the image read-only into a busybox
container (`--mount type=image`, the same thing a pod does), checks the sums, runs `chisel --version`, checks
that the mount is read-only, and writes `$WORK/chisel-1.12.0.oci.tar` (4.7 MB). Push:

```
skopeo copy oci-archive:$WORK/chisel-1.12.0.oci.tar docker://<registry>/<project>/chisel:1.12.0
```

## Test on the cluster

`test-pod.yaml`: a busybox pod (runtime class gVisor) that mounts the image at `/opt/agent-bin/chisel` and
prints:

```
chisel: OK
LICENSE: OK
1.12.0
ok: image volume mounted read-only, binary runs
```

Needs image volume support in the cluster (Kubernetes `ImageVolume` feature on API server and kubelets, and a
container runtime that implements it). Whether image volumes work under gVisor is the main thing to find out.

## Open points for the admin

1. Which sandboxes get the mounts: all on `opensandbox-gvisor`, one tenant, or on request? With all, a
   registry outage would block every pod start.
2. Pull policy and moving tags: does a re-pushed tag reach nodes that already hold the old image?
3. Registry and project to push to, who may push; a public project needs no pull secret.
4. Mount paths in the pod (proposal: `/opt/agent-bin/<binary>/`).
