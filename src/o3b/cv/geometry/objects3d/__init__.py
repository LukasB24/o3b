import importlib
import logging
import pkgutil

logger = logging.getLogger(__name__)

# Plugins are skipped when their optional deps are missing (e.g. nvdiffrast).
# The traceback is logged rather than swallowed: without it a genuine bug inside
# a plugin is indistinguishable from an absent optional dependency.
discovered_plugins = {}
for finder, name, ispkg in pkgutil.iter_modules(__path__, __name__ + "."):
    if ispkg:
        try:
            discovered_plugins[name] = importlib.import_module(name)
        except Exception:
            logger.debug("objects3d: skipping plugin %s", name, exc_info=True)
