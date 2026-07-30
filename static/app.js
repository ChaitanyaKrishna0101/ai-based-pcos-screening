(() => {
  const STEPS = [0, 1, 2, 3, 4]; // 3 (clinical) is skipped when doctor = No
  let currentStep = 0;
  let skipClinical = false;
  let collected = {};   // accumulated payload
  let lastResult = null;

  const panels = document.querySelectorAll(".panel");
  const dots = document.querySelectorAll(".dot");

  function showStep(step) {
    panels.forEach(p => {
      const s = Number(p.dataset.step);
      p.classList.toggle("active", s === step);
    });
    dots.forEach(d => {
      const s = Number(d.dataset.step);
      d.classList.toggle("active", s === step);
      d.classList.toggle("done", s < step);
    });
    currentStep = step;
  }

  function nextAfter(step) {
    if (step === 2 && skipClinical) return 4;
    if (step === 3) return 4;
    return step + 1;
  }
  function prevBefore(step) {
    if (step === 4 && skipClinical) return 2;
    if (step === 2) return 1;
    if (step === 1) return 0;
    return Math.max(0, step - 1);
  }

  document.getElementById("startBtn").addEventListener("click", () => showStep(1));

  document.querySelectorAll("[data-back]").forEach(btn => {
    btn.addEventListener("click", () => showStep(prevBefore(currentStep)));
  });

  // ---- BMI live calc ----
  const basicForm = document.getElementById("basicForm");
  function recalcBMI() {
    const h = parseFloat(basicForm.height_cm.value);
    const w = parseFloat(basicForm.weight_kg.value);
    const out = document.getElementById("bmiOut");
    if (h > 0 && w > 0) {
      const bmi = w / ((h / 100) ** 2);
      out.value = bmi.toFixed(1);
    } else {
      out.value = "";
    }
  }
  basicForm.height_cm.addEventListener("input", recalcBMI);
  basicForm.weight_kg.addEventListener("input", recalcBMI);

  // ---- stress slider ----
  const stressSlider = document.getElementById("stressSlider");
  stressSlider.addEventListener("input", () => {
    document.getElementById("stressVal").textContent = stressSlider.value;
  });

  // ---- skin chips (single-select for simplicity: None clears others) ----
  const skinChips = document.querySelectorAll("#skinChips .chip");
  skinChips.forEach(chip => {
    chip.addEventListener("click", () => {
      if (chip.dataset.value === "None") {
        skinChips.forEach(c => c.classList.remove("selected"));
        chip.classList.add("selected");
      } else {
        document.querySelector('#skinChips .chip[data-value="None"]').classList.remove("selected");
        chip.classList.toggle("selected");
        const anySelected = [...skinChips].some(c => c.classList.contains("selected"));
        if (!anySelected) document.querySelector('#skinChips .chip[data-value="None"]').classList.add("selected");
      }
    });
  });

  function selectedSkinIssues() {
    return [...skinChips].filter(c => c.classList.contains("selected")).map(c => c.dataset.value);
  }

  // ---- step 1 -> step 2 ----
  document.getElementById("toStep2").addEventListener("click", () => {
    if (!basicForm.reportValidity()) return;
    const fd = new FormData(basicForm);
    for (const [k, v] of fd.entries()) collected[k] = v;
    collected.skin_issues = selectedSkinIssues();
    showStep(2);
  });

  // ---- step 2: doctor branch ----
  document.getElementById("docYes").addEventListener("click", () => {
    skipClinical = false;
    collected.consulted_doctor = "Yes";
    showStep(3);
  });
  document.getElementById("docNo").addEventListener("click", () => {
    skipClinical = true;
    collected.consulted_doctor = "No";
    runPrediction();
  });

  // ---- step 3 -> results ----
  const clinicalForm = document.getElementById("clinicalForm");

  // ---- medication yes/no toggle ----
  const medYesNo = document.getElementById("medYesNo");
  const medNameField = document.getElementById("medNameField");
  const medNameInput = document.getElementById("medNameInput");
  const currentMedicationHidden = document.getElementById("currentMedicationHidden");

  medYesNo.addEventListener("change", () => {
    if (medYesNo.value === "Yes") {
      medNameField.style.display = "";
    } else {
      medNameField.style.display = "none";
      medNameInput.value = "";
    }
  });

  document.getElementById("toResult").addEventListener("click", () => {
    currentMedicationHidden.value = (medYesNo.value === "Yes" && medNameInput.value.trim())
      ? medNameInput.value.trim()
      : "None";
    const fd = new FormData(clinicalForm);
    for (const [k, v] of fd.entries()) {
      if (v !== "") collected[k] = v;
    }
    runPrediction();
  });

  // ---- restart ----
  document.getElementById("restartBtn").addEventListener("click", () => {
    collected = {};
    skipClinical = false;
    basicForm.reset();
    clinicalForm.reset();
    medNameField.style.display = "none";
    medNameInput.value = "";
    currentMedicationHidden.value = "None";
    document.getElementById("bmiOut").value = "";
    document.getElementById("stressVal").textContent = "5";
    skinChips.forEach(c => c.classList.remove("selected"));
    document.querySelector('#skinChips .chip[data-value="None"]').classList.add("selected");
    document.getElementById("resultContent").classList.add("hidden");
    document.getElementById("resultError").classList.add("hidden");
    document.getElementById("resultLoading").classList.remove("hidden");
    showStep(0);
  });

  // ---- run prediction ----
  async function runPrediction() {
    showStep(4);
    document.getElementById("resultLoading").classList.remove("hidden");
    document.getElementById("resultContent").classList.add("hidden");
    document.getElementById("resultError").classList.add("hidden");

    try {
      const res = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(collected),
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || "Prediction failed");
      lastResult = data;
      renderResult(data);
    } catch (err) {
      document.getElementById("resultLoading").classList.add("hidden");
      const box = document.getElementById("resultError");
      box.textContent = "Something went wrong: " + err.message;
      box.classList.remove("hidden");
    }
  }

  function renderResult(data) {
    document.getElementById("resultLoading").classList.add("hidden");
    const content = document.getElementById("resultContent");
    content.classList.remove("hidden");

    // gauge
    const pct = Math.round(data.probability * 100);
    document.getElementById("gaugePct").textContent = pct + "%";
    const circumference = 251; // matches path length approx for the arc
    const offset = circumference - (circumference * data.probability);
    const fillEl = document.getElementById("gaugeFill");
    const tierColor = { 1: "#2f9e6b", 2: "#d97706", 3: "#dc4b4b" }[data.tier] || "#6d2e93";
    fillEl.style.stroke = tierColor;
    requestAnimationFrame(() => { fillEl.style.strokeDashoffset = offset; });

    // tier block
    const badge = document.getElementById("tierBadge");
    badge.textContent = "Tier " + data.tier;
    badge.className = "tier-badge t" + data.tier;
    document.getElementById("tierLabel").textContent = data.label;
    document.getElementById("tierReason").textContent = data.tier_reason || "";

    // factors chart
    const chart = document.getElementById("factorsChart");
    chart.innerHTML = "";
    const maxAbs = Math.max(...data.top_factors.map(f => Math.abs(f.shap_value)), 0.0001);
    data.top_factors.forEach(f => {
      const row = document.createElement("div");
      row.className = "factor-row";
      const pctWidth = Math.round((Math.abs(f.shap_value) / maxAbs) * 100);
      const dir = f.direction === "increases_risk" ? "up" : "down";
      row.innerHTML = `
        <span>${f.label}</span>
        <span class="factor-track"><span class="factor-bar ${dir}" style="width:0%"></span></span>
        <span>${f.direction === "increases_risk" ? "↑ risk" : "↓ risk"}</span>
      `;
      chart.appendChild(row);
      requestAnimationFrame(() => {
        row.querySelector(".factor-bar").style.width = pctWidth + "%";
      });
    });
    if (data.top_factors.length === 0) {
      chart.innerHTML = "<p style='color:var(--muted);font-size:.85rem'>Factor breakdown unavailable for this run.</p>";
    }

    // advice cards
    const grid = document.getElementById("adviceGrid");
    const a = data.advice || {};
    const cards = [
      { title: "Summary", html: `<p>${a.summary || ""}</p>`, full: true },
      { title: "Lifestyle", list: a.lifestyle },
      { title: "Ayurvedic / natural support", list: a.ayurvedic },
      { title: "Mental wellbeing", list: a.mental_wellbeing },
      { title: "Avoid", list: a.avoid },
      { title: "Doctor action", html: `<p>${a.doctor_action || ""}</p>`, full: true },
    ];
    grid.innerHTML = "";
    cards.forEach(c => {
      if (!c.html && (!c.list || c.list.length === 0)) return;
      const div = document.createElement("div");
      div.className = "advice-card" + (c.full ? " full" : "");
      const body = c.html || ("<ul>" + c.list.map(i => `<li>${i}</li>`).join("") + "</ul>");
      div.innerHTML = `<h4>${c.title}</h4>${body}`;
      grid.appendChild(div);
    });

    // lab suggestions
    const labBlock = document.getElementById("labBlock");
    if (data.lab_suggestions && data.lab_suggestions.length) {
      labBlock.classList.remove("hidden");
      document.getElementById("labList").innerHTML = data.lab_suggestions.map(t => `<li>${t}</li>`).join("");
    } else {
      labBlock.classList.add("hidden");
    }
  }

  // ---- PDF download ----
  document.getElementById("pdfBtn").addEventListener("click", async () => {
    if (!lastResult) return;
    const btn = document.getElementById("pdfBtn");
    const original = btn.textContent;
    btn.textContent = "Preparing PDF…";
    btn.disabled = true;
    try {
      const res = await fetch("/api/pdf", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(lastResult),
      });
      if (!res.ok) throw new Error("PDF generation failed");
      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url;
      a.download = "pcos_screening_report.pdf";
      document.body.appendChild(a);
      a.click();
      a.remove();
      URL.revokeObjectURL(url);
    } catch (err) {
      alert("Couldn't generate the PDF: " + err.message);
    } finally {
      btn.textContent = original;
      btn.disabled = false;
    }
  });

  showStep(0);
})();
