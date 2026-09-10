"""Process-wide hook used only by the fingerprinted-teacher SkyRL launcher."""

from __future__ import annotations

import os

if os.environ.get("SKYRL_ADFP_TEACHER_ENABLED", "0") == "1":
    from scripts.skyrl_adfp_teacher_patch import install_adfp_teacher_patch

    install_adfp_teacher_patch()
