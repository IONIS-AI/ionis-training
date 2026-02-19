Name:           ionis-training
Version:        3.1.3
Release:        1%{?dist}
Summary:        IONIS training and analysis scripts

License:        GPL-3.0-or-later
URL:            https://github.com/IONIS-AI/ionis-training
Source0:        https://github.com/IONIS-AI/%{name}/archive/v%{version}.tar.gz

BuildArch:      noarch

Obsoletes:      ki7mt-ai-lab-training < 3.0.0
Provides:       ki7mt-ai-lab-training

Requires:       python3 >= 3.9
Requires:       python3-pip
Requires:       ionis-core >= 3.0.0

%description
IONIS (Ionospheric Neural Inference System) training and analysis scripts.
PyTorch-based model predicting HF SNR from WSPR and solar features using
IonisGate architecture (V20 production).

Scripts:
  - coverage_heatmap.py:      Global coverage visualization from ClickHouse
  - signature_search.py:      kNN signature search layer (93M signatures)
  - voacap_batch_runner.py:   VOACAP batch runner for validation
  - voacap_quality_test.py:   VOACAP quality-of-prediction test

# ── validate subpackage ─────────────────────────────────────────────────────

%package validate
Summary:        IONIS V20 Model Validation Suite
Requires:       python3 >= 3.9

%description validate
Standalone validation suite for the IONIS V20 HF propagation model.
Includes 62-test automated battery, custom path testing, and single-prediction
CLI. No ClickHouse or training infrastructure required.

Install dependencies:
  pip3 install -r /usr/share/ionis-training/requirements-validate.txt

Commands:
  ionis-validate test              Run 62-test validation suite
  ionis-validate predict [args]    Predict a single HF path
  ionis-validate custom <file>     Batch custom path tests from JSON
  ionis-validate report [opts]     Generate a beta test report for GitHub Issues
  ionis-validate info              Show model and system information

# ── build ────────────────────────────────────────────────────────────────────

%prep
%autosetup -n %{name}-%{version}

%build
# Nothing to build - Python scripts

%install
# Base package: legacy scripts
install -d %{buildroot}%{_datadir}/%{name}/scripts
install -d %{buildroot}%{_datadir}/%{name}/models

for script in scripts/*.py; do
    install -m 644 "$script" %{buildroot}%{_datadir}/%{name}/scripts/
done



# Validate subpackage
install -d %{buildroot}%{_bindir}
install -d %{buildroot}%{_datadir}/%{name}/versions/v20
install -d %{buildroot}%{_datadir}/%{name}/versions/v20/tests

install -m 755 bin/ionis-validate %{buildroot}%{_bindir}/

install -m 644 versions/v20/model.py %{buildroot}%{_datadir}/%{name}/versions/v20/
install -m 644 versions/v20/config_v20.json %{buildroot}%{_datadir}/%{name}/versions/v20/
install -m 644 versions/v20/ionis_v20.safetensors %{buildroot}%{_datadir}/%{name}/versions/v20/
install -m 644 versions/v20/ionis_v20_meta.json %{buildroot}%{_datadir}/%{name}/versions/v20/

for test_script in versions/v20/tests/*.py; do
    install -m 644 "$test_script" %{buildroot}%{_datadir}/%{name}/versions/v20/tests/
done

install -m 644 requirements-validate.txt %{buildroot}%{_datadir}/%{name}/

# ── file lists ───────────────────────────────────────────────────────────────

%files
%license COPYING
%doc README.md
%dir %{_datadir}/%{name}
%dir %{_datadir}/%{name}/scripts
%dir %{_datadir}/%{name}/models
%{_datadir}/%{name}/scripts/*.py

%files validate
%license COPYING
%{_bindir}/ionis-validate
%dir %{_datadir}/%{name}/versions
%dir %{_datadir}/%{name}/versions/v20
%dir %{_datadir}/%{name}/versions/v20/tests
%{_datadir}/%{name}/versions/v20/model.py
%{_datadir}/%{name}/versions/v20/config_v20.json
%{_datadir}/%{name}/versions/v20/ionis_v20.safetensors
%{_datadir}/%{name}/versions/v20/ionis_v20_meta.json
%{_datadir}/%{name}/versions/v20/tests/*.py
%{_datadir}/%{name}/requirements-validate.txt

# ── changelog ────────────────────────────────────────────────────────────────

%changelog
* Thu Feb 19 2026 Greg Beam <ki7mt@yahoo.com> - 3.2.0-1
- Fix IonisV12Gate → IonisGate in v20 meta.json (class name cleanup)
- Fix v20 README: .pth → .safetensors checkpoint reference
- Update spec description: replace stale V2-era scripts with current inventory
- Archive broken scripts (dashboard.py, predict.py, quality_test_ionis.py)
- Archive stale files (GOAL.md, Modelfile, v10_final.log)
- Remove shipped Modelfile from RPM (Ollama config, not training artifact)
- Remove all .pyc files (generated at runtime)

* Mon Feb 17 2026 Greg Beam <ki7mt@yahoo.com> - 3.1.3-1
- Remove adif command from validate subpackage description

* Mon Feb 17 2026 Greg Beam <ki7mt@yahoo.com> - 3.1.2-1
- Security: replace pickle checkpoint (.pth) with safetensors format
- Update spec and debian/rules for ionis_v20.safetensors

* Mon Feb 17 2026 Greg Beam <ki7mt@yahoo.com> - 3.1.1-1
- Security hardening: weights_only=True for all checkpoint loading
- Add Debian packaging for Launchpad PPA (debian/ directory)

* Mon Feb 16 2026 Greg Beam <ki7mt@yahoo.com> - 3.1.0-1
- Add ionis-training-validate subpackage for beta testing
- Extract model.py from train_common.py (zero ClickHouse dependency)
- Add ionis-validate CLI: test, predict, custom, info commands
- Fix device selection: CUDA > MPS > CPU (universal)
- Fix run_all.py Python discovery (use sys.executable)
- Ship V20 checkpoint (808 KB) in validate RPM

* Mon Feb 16 2026 Greg Beam <ki7mt@yahoo.com> - 3.0.1-1
- Complete V20 test suite: 62 tests across 8 groups (TST-100 through TST-800)
- Replace hardcoded paths with $IONIS_WORKSPACE throughout
- Clean up legacy V12/V16 naming — class is IonisGate, period

* Fri Feb 13 2026 Greg Beam <ki7mt@yahoo.com> - 3.0.0-1
- Rename package: ki7mt-ai-lab-training → ionis-training
- Move to IONIS-AI GitHub org
- Require ionis-core >= 3.0.0
- Add Obsoletes/Provides for seamless RPM upgrade

* Thu Feb 12 2026 Greg Beam <ki7mt@yahoo.com> - 2.4.0-2
- Remove .pth checkpoints from package (moved to ZFS archive-pool/ionis-models)
- RPM now ships scripts only — model artifacts managed separately

* Wed Feb 11 2026 Greg Beam <ki7mt@yahoo.com> - 2.4.0-1
- V20 production release
- Update description: ResidualBlock → IonisGate (V20 production)

* Sun Feb 08 2026 Greg Beam <ki7mt@yahoo.com> - 2.3.1-1
- Medallion architecture: gold_* table references
- Align version across all lab packages at 2.3.1

* Sat Feb 07 2026 Greg Beam <ki7mt@yahoo.com> - 2.3.0-1
- Align version across all lab packages at 2.3.0

* Wed Feb 04 2026 Greg Beam <ki7mt@yahoo.com> - 2.2.0-1
- Align version across all lab packages at 2.2.0 for Phase 4.1

* Tue Feb 03 2026 Greg Beam <ki7mt@yahoo.com> - 2.1.0-1
- Initial packaging for COPR
- IONIS V2 training scripts and sensitivity analysis
- Align version across all lab packages at 2.1.0
