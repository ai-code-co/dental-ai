import React, { useMemo, useState } from "react";

const API_BASE = "http://localhost:8000";

export default function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState("");
  const [result, setResult] = useState(null);

  const [loadingAnalyze, setLoadingAnalyze] = useState(false);
  const [loadingPdf, setLoadingPdf] = useState(false);
  const [error, setError] = useState("");

  const canAnalyze = !!selectedFile && !loadingAnalyze;

  const statusPill = useMemo(() => {
    if (!result) return null;
    const isOptimal = result.status === "Optimal";
    const base = "px-3 py-1 rounded-full text-[11px] font-bold tracking-wide";
    return (
      <span className={`${base} ${isOptimal ? "bg-green-100 text-green-700" : "bg-red-100 text-red-700"}`}>
        {result.status.toUpperCase()}
      </span>
    );
  }, [result]);

  const onPickFile = (file) => {
    setError("");
    setResult(null);
    setSelectedFile(file);
    const url = URL.createObjectURL(file);
    setPreviewUrl(url);
  };

  const handleFileInput = (e) => {
    const file = e.target.files?.[0];
    if (file) onPickFile(file);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file) onPickFile(file);
  };

  const handleAnalyze = async () => {
    if (!selectedFile) return;
    setError("");
    setLoadingAnalyze(true);

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const res = await fetch(`${API_BASE}/analyze`, { method: "POST", body: formData });
      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg?.detail || "Analyze failed");
      }
      const data = await res.json();
      setResult(data);
    } catch (e) {
      setError(e.message || "Backend offline / request failed");
    } finally {
      setLoadingAnalyze(false);
    }
  };

  const downloadReport = async () => {
    if (!result) return;
    setError("");
    setLoadingPdf(true);

    try {
      const res = await fetch(`${API_BASE}/generate-pdf`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(result),
      });

      if (!res.ok) {
        const msg = await res.json().catch(() => ({}));
        throw new Error(msg?.detail || "PDF generation failed");
      }

      const blob = await res.blob();
      const url = window.URL.createObjectURL(blob);

      const link = document.createElement("a");
      link.href = url;
      link.download = "Patient_Validation_Report.pdf";
      link.click();

      window.URL.revokeObjectURL(url);
    } catch (e) {
      setError(e.message || "PDF download failed");
    } finally {
      setLoadingPdf(false);
    }
  };

  return (
    <div className="min-h-screen bg-[#F4F7FA] font-sans">
      <nav className="bg-[#002B5B] p-6 shadow-xl flex justify-between items-center text-white">
        <div className="flex items-center space-x-3">
          <span className="text-3xl">🦷</span>
          <h1 className="text-2xl font-bold">
            DentalVision <span className="font-light">Validation</span>
          </h1>
        </div>
        <div className="text-xs opacity-70">
          LOCAL DEV • API: {API_BASE}
        </div>
      </nav>

      <main className="max-w-7xl mx-auto py-12 px-6 grid grid-cols-12 gap-10">
        {/* Left Column */}
        <div className="col-span-12 lg:col-span-4 space-y-6">
          <div className="bg-white p-8 rounded-3xl shadow-sm border border-slate-200">
            <h2 className="text-slate-800 font-bold text-xl mb-2">Upload Profile Image</h2>
            <p className="text-sm text-slate-500 mb-6">
              Upload a right/left profile image with the positioning system visible.
            </p>

            <div
              onDragOver={(e) => e.preventDefault()}
              onDrop={handleDrop}
              className="border-2 border-dashed border-blue-200 p-8 rounded-2xl text-center bg-blue-50 hover:bg-blue-100 transition-all cursor-pointer"
              onClick={() => document.getElementById("fileIn").click()}
            >
              <input
                type="file"
                id="fileIn"
                className="hidden"
                accept="image/*"
                onChange={handleFileInput}
              />
              <span className="text-5xl block mb-3">📸</span>
              <div className="text-sm text-blue-700 font-bold">
                {selectedFile ? selectedFile.name : "Click or drag & drop an image"}
              </div>
              <div className="text-xs text-blue-700/70 mt-1">JPG / PNG</div>
            </div>

            {previewUrl && (
              <div className="mt-6 rounded-2xl overflow-hidden border border-slate-200">
                <img src={previewUrl} alt="preview" className="w-full" />
              </div>
            )}

            <button
              onClick={handleAnalyze}
              disabled={!canAnalyze}
              className="w-full mt-6 bg-[#007BFF] hover:bg-blue-700 text-white py-4 rounded-2xl font-black shadow-xl transition-all disabled:bg-slate-300"
            >
              {loadingAnalyze ? "ANALYZING..." : "RUN VALIDATION"}
            </button>

            {error && (
              <div className="mt-4 bg-red-50 border border-red-200 text-red-700 text-sm p-3 rounded-xl">
                {error}
              </div>
            )}
          </div>

          {result && (
            <button
              onClick={downloadReport}
              disabled={loadingPdf}
              className="w-full bg-emerald-600 hover:bg-emerald-700 text-white py-4 rounded-2xl font-bold shadow-lg flex items-center justify-center space-x-2 disabled:bg-emerald-300"
            >
              <span>📄</span>
              <span>{loadingPdf ? "GENERATING PDF..." : "DOWNLOAD PDF REPORT"}</span>
            </button>
          )}
        </div>

        {/* Right Column */}
        <div className="col-span-12 lg:col-span-8 bg-white p-10 rounded-3xl shadow-sm border border-slate-200">
          {!result ? (
            <div className="h-full flex flex-col items-center justify-center text-slate-300 py-20">
              <span className="text-8xl mb-6">🔬</span>
              <p className="text-xl font-medium">Upload an image to begin validation</p>
              <p className="text-sm mt-2">You’ll get angle, orientation, and an annotated preview.</p>
            </div>
          ) : (
            <div className="space-y-8">
              <div className="flex justify-between items-center border-b pb-6">
                <div>
                  <h2 className="text-2xl font-black text-[#002B5B]">Diagnostic Overview</h2>
                  <p className="text-sm text-slate-500 mt-1">
                    Tolerance: ±{result.tolerance_deg}°
                  </p>
                </div>
                {statusPill}
              </div>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className="bg-slate-50 p-6 rounded-2xl">
                  <p className="text-[10px] text-slate-400 font-bold uppercase mb-2 tracking-widest">
                    Orientation
                  </p>
                  <p className="text-xl font-bold text-slate-700">{result.orientation}</p>
                </div>

                <div className="bg-slate-50 p-6 rounded-2xl">
                  <p className="text-[10px] text-slate-400 font-bold uppercase mb-2 tracking-widest">
                    Baseline Angle
                  </p>
                  <p className="text-3xl font-black text-blue-600">{result.angle}°</p>
                </div>

                <div className="bg-slate-50 p-6 rounded-2xl">
                  <p className="text-[10px] text-slate-400 font-bold uppercase mb-2 tracking-widest">
                    Scale Reference
                  </p>
                  <p className="text-lg font-bold text-slate-700">{result.scale}</p>
                </div>
              </div>

              <div className="bg-blue-50 border-l-4 border-blue-500 p-6 rounded-2xl">
                <p className="text-[10px] text-blue-600 font-bold uppercase mb-2">
                  Clinical Observation (AI)
                </p>
                <p className="text-blue-900 italic font-medium">“{result.note}”</p>
              </div>

              <div className="relative rounded-3xl overflow-hidden shadow-xl border border-slate-200">
                <img src={result.image} alt="AI Mapping" className="w-full" />
                <div className="absolute top-4 left-4 bg-black/60 text-white text-[10px] px-3 py-1 rounded-full font-bold">
                  LANDMARK LINE: EAR ↔ EYE
                </div>
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}
