"""Historical experiment implementations kept outside the runtime package.

Nothing under :mod:`archive` is a production entrypoint.  Archived modules may
import the active runtime to reproduce an old experiment, but active modules
must not import archive code except through the explicitly deprecated legacy
compatibility shims.
"""

