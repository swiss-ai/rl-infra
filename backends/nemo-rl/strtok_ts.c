// strtok() with per-thread state, preloaded into the job's processes (SITE_PRELOAD).
//
// libfabric's CXI provider (prov/cxi/src/cxip_nic.c, up to at least 2.6.0 and main of
// 2026-10-28) parses SLINGSHOT_DEVICES and SLINGSHOT_SVC_IDS with strtok() on every
// fi_getinfo(), and glibc's strtok() keeps its position in one process-wide variable.
// UCCL's proxy threads call fi_getinfo() at the same time, so one thread's parse can
// continue in another thread's string: the NIC is then not found in SLINGSHOT_DEVICES (or
// the process crashes). libfabric falls back to looking up a service by user ID, which
// fails inside a pyxis container (uid 0), leaves the service ID at 0, and fi_domain()
// returns -FI_ENOSYS ("Function not implemented").
#include <string.h>

static __thread char *saved;

char *strtok(char *s, const char *delim) {
    return strtok_r(s, delim, &saved);
}
