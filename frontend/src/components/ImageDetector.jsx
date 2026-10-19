"use client";

import React, { useState, useRef, useEffect } from "react";

export default function ImageDetector({
  confidence,
  iouThreshold,
  selectedClasses,
  showLabels,
  showConfidence,
  showCoordinates,
  showGridGuide,
  samples = [],
  usePyramid = true,
  pyramidType = "both",
}) {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreviewUrl, setImagePreviewUrl] = useState(null);
  const [detectionResult, setDetectionResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [hoveredBoxIndex, setHoveredBoxIndex] = useState(null);

  const canvasRef = useRef(null);
  const imageElementRef = useRef(null);
  const fileInputRef = useRef(null);

  // Load first sample on mount if available
  useEffect(() => {
    if (samples.length > 0 && !imagePreviewUrl) {
      loadSample(samples[0]);
    }
  }, [samples]);

  const loadSample = async (sample) => {
    try {
      setLoading(true);
      setError(null);
      setImagePreviewUrl(sample.url);

      // Fetch the sample image blob and predict
      const res = await fetch(sample.url);
      const blob = await res.blob();
      const file = new File([blob], sample.filename, { type: "image/jpeg" });
      setSelectedImage(file);
      runInference(file);
    } catch (err) {
      setError("Failed to load sample image: " + err.message);
      setLoading(false);
    }
  };

  const handleFileChange = (e) => {
    const file = e.target.files?.[0];
    if (file) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setImagePreviewUrl(url);
      runInference(file);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const file = e.dataTransfer.files?.[0];
    if (file && file.type.startsWith("image/")) {
      setSelectedImage(file);
      const url = URL.createObjectURL(file);
      setImagePreviewUrl(url);
      runInference(file);
    }
  };

  const runInference = async (fileToInfer) => {
    setLoading(true);
    setError(null);
    try {
      const formData = new FormData();
      formData.append("file", fileToInfer);
      formData.append("confidence_threshold", confidence);
      formData.append("iou_threshold", iouThreshold);
      formData.append("use_pyramid", usePyramid ? "true" : "false");
      formData.append("pyramid_type", pyramidType);
      if (selectedClasses && selectedClasses.length > 0) {
        formData.append("allowed_classes", JSON.stringify(selectedClasses));
      }

      const res = await fetch("/api/predict/image", {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errorData = await res.json().catch(() => ({}));
        throw new Error(errorData.detail || `Server error: ${res.status}`);
      }

      const data = await res.json();
      setDetectionResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  // Re-run inference when criteria sliders change and we have an image
  useEffect(() => {
    if (selectedImage) {
      const timer = setTimeout(() => runInference(selectedImage), 350);
      return () => clearTimeout(timer);
    }
  }, [confidence, iouThreshold, selectedClasses, usePyramid, pyramidType]);

  // Canvas drawing effect
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !imagePreviewUrl) return;

    const ctx = canvas.getContext("2d");
    const img = new Image();
    img.crossOrigin = "anonymous";
    img.src = imagePreviewUrl;

    img.onload = () => {
      imageElementRef.current = img;

      // Set canvas internal resolution to original image resolution
      canvas.width = img.naturalWidth;
      canvas.height = img.naturalHeight;

      // Draw original image
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(img, 0, 0);

      // Optional anchor grid overlay (representing OverFeat sliding windows)
      if (showGridGuide && detectionResult?.grid_size) {
        const { height: Gh, width: Gw } = detectionResult.grid_size;
        ctx.strokeStyle = "rgba(56, 189, 248, 0.15)";
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);

        for (let y = 0; y < Gh; y++) {
          for (let x = 0; x < Gw; x++) {
            const ax = x * 32;
            const ay = y * 32;
            ctx.strokeRect(ax, ay, 224, 224);
          }
        }
        ctx.setLineDash([]);
      }

      // Draw bounding boxes
      if (detectionResult && detectionResult.detections) {
        detectionResult.detections.forEach((det, idx) => {
          const [xmin, ymin, xmax, ymax] = det.box;
          const w = xmax - xmin;
          const h = ymax - ymin;
          const color = det.color || "#38bdf8";
          const isHovered = hoveredBoxIndex === idx;

          // Box fill
          ctx.fillStyle = isHovered ? `${color}40` : `${color}20`;
          ctx.fillRect(xmin, ymin, w, h);

          // Box border
          ctx.strokeStyle = color;
          ctx.lineWidth = isHovered ? 4 : 2.5;
          ctx.strokeRect(xmin, ymin, w, h);

          // Corners emphasis
          const cornerLen = Math.min(16, w / 4, h / 4);
          ctx.lineWidth = isHovered ? 5 : 3.5;
          // top-left
          ctx.beginPath();
          ctx.moveTo(xmin, ymin + cornerLen);
          ctx.lineTo(xmin, ymin);
          ctx.lineTo(xmin + cornerLen, ymin);
          ctx.stroke();
          // top-right
          ctx.beginPath();
          ctx.moveTo(xmax - cornerLen, ymin);
          ctx.lineTo(xmax, ymin);
          ctx.lineTo(xmax, ymin + cornerLen);
          ctx.stroke();
          // bottom-left
          ctx.beginPath();
          ctx.moveTo(xmin, ymax - cornerLen);
          ctx.lineTo(xmin, ymax);
          ctx.lineTo(xmin + cornerLen, ymax);
          ctx.stroke();
          // bottom-right
          ctx.beginPath();
          ctx.moveTo(xmax - cornerLen, ymax);
          ctx.lineTo(xmax, ymax);
          ctx.lineTo(xmax, ymax - cornerLen);
          ctx.stroke();

          // Pill tag with label and confidence
          if (showLabels || showConfidence) {
            let labelText = "";
            if (showLabels) labelText += det.class_name.toUpperCase();
            if (showConfidence) {
              const confPct = Math.round(det.confidence * 100);
              labelText += (showLabels ? " • " : "") + `${confPct}%`;
            }

            ctx.font = "bold 13px 'Inter', sans-serif";
            const textMetrics = ctx.measureText(labelText);
            const tagHeight = 22;
            const tagWidth = textMetrics.width + 16;
            const tagY = Math.max(0, ymin - tagHeight - 2);

            // Tag background
            ctx.fillStyle = color;
            ctx.beginPath();
            ctx.roundRect(xmin, tagY, tagWidth, tagHeight, [4, 4, 0, 0]);
            ctx.fill();

            // Tag text
            ctx.fillStyle = "#ffffff";
            ctx.textBaseline = "middle";
            ctx.fillText(labelText, xmin + 8, tagY + tagHeight / 2);
          }

          // Optional coordinates tag
          if (showCoordinates) {
            const coordText = `${Math.round(w)}x${Math.round(h)}px`;
            ctx.font = "10px monospace";
            ctx.fillStyle = "rgba(0, 0, 0, 0.65)";
            ctx.fillRect(xmin, ymax - 18, 70, 18);
            ctx.fillStyle = "#f8fafc";
            ctx.fillText(coordText, xmin + 4, ymax - 5);
          }
        });
      }
    };
  }, [
    imagePreviewUrl,
    detectionResult,
    showLabels,
    showConfidence,
    showCoordinates,
    showGridGuide,
    hoveredBoxIndex,
  ]);

  const downloadAnnotatedImage = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const link = document.createElement("a");
    link.download = `overfeat_detection_${Date.now()}.png`;
    link.href = canvas.toDataURL("image/png");
    link.click();
  };

  const exportJSON = () => {
    if (!detectionResult) return;
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(detectionResult, null, 2));
    const link = document.createElement("a");
    link.download = `detections_${Date.now()}.json`;
    link.href = dataStr;
    link.click();
  };

  return (
    <div style={styles.container}>
      {/* Top Banner / Sample Gallery */}
      <div style={styles.sampleBar}>
        <span style={styles.sampleLabel}>VOC 2007 Preset Samples:</span>
        <div style={styles.samplesList}>
          {samples.map((s) => (
            <button
              key={s.id}
              style={styles.sampleButton}
              onClick={() => loadSample(s)}
            >
              <span style={styles.sampleDot} />
              <span>{s.label}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Main workspace */}
      <div style={styles.workspace}>
        {/* Left: Canvas Area */}
        <div style={styles.canvasContainer}>
          {/* Dropzone / Upload overlay if no image or to replace */}
          <div
            style={styles.uploadStrip}
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept="image/*"
              style={{ display: "none" }}
              onChange={handleFileChange}
            />
            <button
              style={styles.browseButton}
              onClick={() => fileInputRef.current?.click()}
            >
              <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                <polyline points="17 8 12 3 7 8" />
                <line x1="12" y1="3" x2="12" y2="15" />
              </svg>
              <span>Upload Custom Image</span>
            </button>
            <span style={styles.dragNote}>or drag & drop anywhere here</span>

            {/* Quick Actions */}
            {detectionResult && (
              <div style={styles.actionGroup}>
                <button style={styles.actionBtn} onClick={downloadAnnotatedImage} title="Download Annotated Image">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7 10 12 15 17 10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  <span>Save Image</span>
                </button>
                <button style={styles.actionBtn} onClick={exportJSON} title="Export JSON detections">
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="16 18 22 12 16 6"/>
                    <polyline points="8 6 2 12 8 18"/>
                  </svg>
                  <span>JSON</span>
                </button>
              </div>
            )}
          </div>

          {/* Canvas Wrapper */}
          <div style={styles.canvasFrame}>
            {loading && (
              <div style={styles.loadingOverlay}>
                <div style={styles.spinner} />
                <span>Running OverFeat CNN Inference...</span>
              </div>
            )}
            {error && (
              <div style={styles.errorAlert}>
                <span>{error}</span>
              </div>
            )}
            <canvas ref={canvasRef} style={styles.canvas} />
          </div>
        </div>

        {/* Right: Results Panel */}
        <div style={styles.resultsPanel}>
          {/* Image Pyramid Banner if enabled */}
          {detectionResult?.pyramid_info?.enabled && (
            <div style={styles.pyramidBanner}>
              <div style={styles.pyramidBannerLeft}>
                <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2">
                  <polygon points="12 2 2 22 22 22" />
                  <line x1="7" y1="12" x2="17" y2="12" />
                </svg>
                <span style={styles.pyramidBannerText}>
                  Image Pyramid: <strong>{detectionResult.pyramid_info.num_scales} scales</strong> (
                  {detectionResult.pyramid_info.scales_used?.map((s) => `${s}x`).join(", ")})
                </span>
              </div>
              <span style={styles.pyramidBannerPill}>
                {detectionResult.pyramid_info.pyramid_type?.toUpperCase()}
              </span>
            </div>
          )}

          {/* Metrics summary */}
          <div style={styles.metricsGrid}>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Detections</span>
              <span style={styles.metricValueCyan}>
                {detectionResult?.num_detections ?? 0}
              </span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Inference</span>
              <span style={styles.metricValueEmerald}>
                {detectionResult?.timings_ms?.inference ?? 0} ms
              </span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Total Latency</span>
              <span style={styles.metricValueViolet}>
                {detectionResult?.timings_ms?.total ?? 0} ms
              </span>
            </div>
            <div style={styles.metricCard}>
              <span style={styles.metricLabel}>Model Speed</span>
              <span style={styles.metricValueAmber}>
                {detectionResult?.timings_ms?.fps ?? 0} FPS
              </span>
            </div>
          </div>

          {/* Detections List */}
          <div style={styles.detectionsBox}>
            <div style={styles.detectionsHeader}>
              <span style={styles.detectionsTitle}>Detected Objects</span>
              <span style={styles.detectionsCount}>
                Criteria: {Math.round(confidence * 100)}% conf • {Math.round(iouThreshold * 100)}% IoU
              </span>
            </div>

            {(!detectionResult || detectionResult.detections.length === 0) ? (
              <div style={styles.emptyState}>
                {loading ? (
                  <span>Processing image...</span>
                ) : (
                  <span>No bounding boxes satisfy criteria. Try lowering the confidence slider.</span>
                )}
              </div>
            ) : (
              <div style={styles.detectionsList}>
                {detectionResult.detections.map((det, idx) => {
                  const isHovered = hoveredBoxIndex === idx;
                  const color = det.color || "#38bdf8";
                  return (
                    <div
                      key={idx}
                      style={{
                        ...styles.detectionCard,
                        borderColor: isHovered ? color : "rgba(255, 255, 255, 0.08)",
                        backgroundColor: isHovered
                          ? `${color}18`
                          : "rgba(18, 24, 38, 0.6)",
                      }}
                      onMouseEnter={() => setHoveredBoxIndex(idx)}
                      onMouseLeave={() => setHoveredBoxIndex(null)}
                    >
                      <div style={styles.detTop}>
                        <div style={styles.detClassGroup}>
                          <span
                            style={{
                              ...styles.detColorIndicator,
                              backgroundColor: color,
                            }}
                          />
                          <strong style={styles.detClassName}>
                            {det.class_name}
                          </strong>
                          {det.scale != null && (
                            <span
                              style={{
                                ...styles.detScaleBadge,
                                ...(det.scale_role === "up"
                                  ? styles.detScaleBadgeUp
                                  : det.scale_role === "down"
                                  ? styles.detScaleBadgeDown
                                  : styles.detScaleBadgeBase),
                              }}
                              title={`Detected at ${det.scale}x scale (${det.scale_role})`}
                            >
                              {det.scale}x {det.scale_role}
                            </span>
                          )}
                        </div>
                        <span style={{ ...styles.detConf, color }}>
                          {Math.round(det.confidence * 100)}%
                        </span>
                      </div>

                      <div style={styles.detCoords}>
                        <span>Box: [{det.box.join(", ")}]</span>
                        <span>Area: {Math.round(det.area)} px²</span>
                      </div>

                      <div style={styles.confBarTrack}>
                        <div
                          style={{
                            ...styles.confBarFill,
                            width: `${Math.round(det.confidence * 100)}%`,
                            backgroundColor: color,
                          }}
                        />
                      </div>
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

const styles = {
  container: {
    display: "flex",
    flexDirection: "column",
    gap: "14px",
    flex: 1,
  },
  sampleBar: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    background: "rgba(18, 24, 38, 0.5)",
    padding: "8px 16px",
    borderRadius: "10px",
    border: "1px solid rgba(255, 255, 255, 0.06)",
    overflowX: "auto",
  },
  sampleLabel: {
    fontSize: "12px",
    fontWeight: "600",
    color: "#94a3b8",
    whiteSpace: "nowrap",
  },
  samplesList: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  sampleButton: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "5px 12px",
    fontSize: "12px",
    fontWeight: "500",
    color: "#e2e8f0",
    background: "rgba(255, 255, 255, 0.04)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    borderRadius: "8px",
    cursor: "pointer",
    whiteSpace: "nowrap",
    transition: "all 0.15s ease",
  },
  sampleDot: {
    width: "6px",
    height: "6px",
    borderRadius: "50%",
    backgroundColor: "#38bdf8",
  },
  workspace: {
    display: "flex",
    gap: "16px",
    flex: 1,
    minHeight: "560px",
  },
  canvasContainer: {
    flex: 1.5,
    display: "flex",
    flexDirection: "column",
    gap: "10px",
    background: "rgba(14, 19, 31, 0.6)",
    borderRadius: "16px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
    overflow: "hidden",
  },
  uploadStrip: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 12px",
    background: "rgba(18, 24, 38, 0.7)",
    borderRadius: "10px",
    border: "1px dashed rgba(255, 255, 255, 0.15)",
  },
  browseButton: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "6px 14px",
    background: "linear-gradient(135deg, #0284c7, #0369a1)",
    color: "#ffffff",
    border: "none",
    borderRadius: "8px",
    fontSize: "12.5px",
    fontWeight: "600",
    cursor: "pointer",
  },
  dragNote: {
    fontSize: "11.5px",
    color: "#64748b",
  },
  actionGroup: {
    display: "flex",
    gap: "6px",
  },
  actionBtn: {
    display: "flex",
    alignItems: "center",
    gap: "5px",
    padding: "5px 10px",
    fontSize: "11.5px",
    fontWeight: "500",
    color: "#cbd5e1",
    background: "rgba(255, 255, 255, 0.06)",
    border: "1px solid rgba(255, 255, 255, 0.1)",
    borderRadius: "6px",
    cursor: "pointer",
  },
  canvasFrame: {
    position: "relative",
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    background: "#080b11",
    borderRadius: "12px",
    border: "1px solid rgba(255, 255, 255, 0.05)",
    overflow: "hidden",
    minHeight: "440px",
  },
  canvas: {
    maxWidth: "100%",
    maxHeight: "100%",
    objectFit: "contain",
    borderRadius: "8px",
  },
  loadingOverlay: {
    position: "absolute",
    inset: 0,
    background: "rgba(8, 11, 17, 0.8)",
    backdropFilter: "blur(4px)",
    display: "flex",
    flexDirection: "column",
    alignItems: "center",
    justifyContent: "center",
    gap: "12px",
    color: "#38bdf8",
    fontSize: "13px",
    zIndex: 10,
  },
  spinner: {
    width: "36px",
    height: "36px",
    border: "3px solid rgba(56, 189, 248, 0.2)",
    borderTopColor: "#38bdf8",
    borderRadius: "50%",
    animation: "spin-slow 0.8s linear infinite",
  },
  errorAlert: {
    position: "absolute",
    top: "14px",
    left: "14px",
    right: "14px",
    background: "rgba(239, 68, 68, 0.2)",
    border: "1px solid rgba(239, 68, 68, 0.4)",
    color: "#fca5a5",
    padding: "10px 14px",
    borderRadius: "8px",
    fontSize: "12px",
    zIndex: 10,
  },
  resultsPanel: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    gap: "14px",
  },
  metricsGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(4, 1fr)",
    gap: "8px",
  },
  metricCard: {
    display: "flex",
    flexDirection: "column",
    gap: "4px",
    background: "rgba(18, 24, 38, 0.7)",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    borderRadius: "10px",
    padding: "10px 12px",
  },
  metricLabel: {
    fontSize: "10.5px",
    fontWeight: "600",
    textTransform: "uppercase",
    letterSpacing: "0.5px",
    color: "#94a3b8",
  },
  metricValueCyan: {
    fontSize: "18px",
    fontWeight: "800",
    color: "#38bdf8",
  },
  metricValueEmerald: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#10b981",
  },
  metricValueViolet: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#a855f7",
  },
  metricValueAmber: {
    fontSize: "16px",
    fontWeight: "700",
    color: "#f59e0b",
  },
  detectionsBox: {
    flex: 1,
    display: "flex",
    flexDirection: "column",
    background: "rgba(14, 19, 31, 0.6)",
    borderRadius: "16px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
    padding: "14px",
    overflow: "hidden",
  },
  detectionsHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: "10px",
    borderBottom: "1px solid rgba(255, 255, 255, 0.06)",
    marginBottom: "10px",
  },
  detectionsTitle: {
    fontSize: "13px",
    fontWeight: "700",
    color: "#e2e8f0",
  },
  detectionsCount: {
    fontSize: "11px",
    color: "#64748b",
  },
  emptyState: {
    flex: 1,
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    textAlign: "center",
    color: "#64748b",
    fontSize: "12.5px",
    padding: "20px",
  },
  detectionsList: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    overflowY: "auto",
    maxHeight: "380px",
    paddingRight: "4px",
  },
  detectionCard: {
    display: "flex",
    flexDirection: "column",
    gap: "6px",
    padding: "10px 12px",
    borderRadius: "10px",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(255, 255, 255, 0.08)",
    transition: "all 0.15s ease",
    cursor: "pointer",
  },
  detTop: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
  },
  detClassGroup: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  detColorIndicator: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
  },
  detClassName: {
    fontSize: "13px",
    textTransform: "capitalize",
    color: "#f8fafc",
  },
  detConf: {
    fontSize: "12.5px",
    fontWeight: "700",
  },
  detCoords: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "10.5px",
    color: "#94a3b8",
    fontFamily: "monospace",
  },
  confBarTrack: {
    width: "100%",
    height: "4px",
    backgroundColor: "rgba(255, 255, 255, 0.08)",
    borderRadius: "2px",
    overflow: "hidden",
  },
  confBarFill: {
    height: "100%",
    borderRadius: "2px",
  },
  pyramidBanner: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "8px 12px",
    borderRadius: "8px",
    background: "rgba(56, 189, 248, 0.08)",
    border: "1px solid rgba(56, 189, 248, 0.22)",
    marginBottom: "12px",
  },
  pyramidBannerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  pyramidBannerText: {
    fontSize: "11.5px",
    color: "#e2e8f0",
  },
  pyramidBannerPill: {
    fontSize: "10px",
    fontWeight: "700",
    padding: "2px 8px",
    borderRadius: "10px",
    background: "rgba(56, 189, 248, 0.2)",
    color: "#38bdf8",
    border: "1px solid rgba(56, 189, 248, 0.3)",
    letterSpacing: "0.5px",
  },
  detScaleBadge: {
    fontSize: "9px",
    fontWeight: "700",
    padding: "1px 5px",
    borderRadius: "4px",
    letterSpacing: "0.3px",
  },
  detScaleBadgeUp: {
    color: "#34d399",
    background: "rgba(52, 211, 153, 0.14)",
    border: "1px solid rgba(52, 211, 153, 0.28)",
  },
  detScaleBadgeDown: {
    color: "#c084fc",
    background: "rgba(192, 132, 252, 0.14)",
    border: "1px solid rgba(192, 132, 252, 0.28)",
  },
  detScaleBadgeBase: {
    color: "#38bdf8",
    background: "rgba(56, 189, 248, 0.14)",
    border: "1px solid rgba(56, 189, 248, 0.28)",
  },
};
