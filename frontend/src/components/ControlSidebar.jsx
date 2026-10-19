"use client";

import React, { useMemo } from "react";

const VOC_CATEGORIES = {
  All: null,
  Vehicles: ["aeroplane", "bicycle", "boat", "bus", "car", "motorbike", "train"],
  Animals: ["bird", "cat", "cow", "dog", "horse", "sheep"],
  Indoor: ["bottle", "chair", "diningtable", "pottedplant", "sofa", "tvmonitor"],
  Person: ["person"],
};

export default function ControlSidebar({
  activeTab,
  fps,
  setFps,
  confidence,
  setConfidence,
  iouThreshold,
  setIouThreshold,
  selectedClasses,
  setSelectedClasses,
  allClasses = [],
  classColors = {},
  showLabels,
  setShowLabels,
  showConfidence,
  setShowConfidence,
  showCoordinates,
  setShowCoordinates,
  showGridGuide,
  setShowGridGuide,
  usePyramid = true,
  setUsePyramid,
  pyramidType = "both",
  setPyramidType,
}) {
  const isStreamMode = activeTab === "camera" || activeTab === "video";

  const toggleClass = (cls) => {
    if (selectedClasses.includes(cls)) {
      setSelectedClasses(selectedClasses.filter((c) => c !== cls));
    } else {
      setSelectedClasses([...selectedClasses, cls]);
    }
  };

  const selectCategory = (cat) => {
    if (cat === "All") {
      setSelectedClasses([...allClasses]);
    } else {
      setSelectedClasses(VOC_CATEGORIES[cat]);
    }
  };

  const isAllSelected = selectedClasses.length === allClasses.length;

  return (
    <aside style={styles.sidebar}>
      <div style={styles.sectionHeader}>
        <span style={styles.sectionTitle}>Processing & Criteria</span>
        <button
          style={styles.resetButton}
          onClick={() => {
            setConfidence(0.25);
            setIouThreshold(0.45);
            setFps(10);
            setSelectedClasses([...allClasses]);
            if (setUsePyramid) setUsePyramid(true);
            if (setPyramidType) setPyramidType("both");
          }}
          title="Reset all settings to defaults"
        >
          Reset
        </button>
      </div>

      {/* Image Pyramid (Multi-Scale: Up + Down scale) */}
      <div
        style={{
          ...styles.card,
          ...(usePyramid ? styles.cardPyramidActive : {}),
        }}
      >
        <div style={styles.controlHeader}>
          <div style={styles.labelWithIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2">
              <polygon points="12 2 2 22 22 22" />
              <line x1="7" y1="12" x2="17" y2="12" />
              <line x1="9.5" y1="7" x2="14.5" y2="7" />
            </svg>
            <label style={styles.controlLabel}>Image Pyramid</label>
          </div>
          <button
            onClick={() => setUsePyramid && setUsePyramid(!usePyramid)}
            style={{
              ...styles.pyramidToggleBtn,
              backgroundColor: usePyramid ? "rgba(56, 189, 248, 0.2)" : "rgba(255, 255, 255, 0.05)",
              color: usePyramid ? "#38bdf8" : "#94a3b8",
              borderColor: usePyramid ? "#38bdf8" : "rgba(255, 255, 255, 0.1)",
            }}
            title={usePyramid ? "Disable multi-scale Image Pyramid" : "Enable multi-scale Image Pyramid"}
          >
            {usePyramid ? "ACTIVE" : "OFF"}
          </button>
        </div>

        <p style={styles.controlHelp}>
          Multi-scale sampling for OverFeat. <strong style={{ color: "#38bdf8" }}>Up-scale</strong> resolves small objects; <strong style={{ color: "#a855f7" }}>Down-scale</strong> captures large objects.
        </p>

        {usePyramid && (
          <div style={{ marginTop: "10px" }}>
            <div style={styles.pyramidModeSelector}>
              <button
                style={{
                  ...styles.pyramidModeChip,
                  ...(pyramidType === "both" ? styles.pyramidModeChipActive : {}),
                }}
                onClick={() => setPyramidType && setPyramidType("both")}
                title="Both Up-scale and Down-scale"
              >
                Both (Up + Down)
              </button>
              <button
                style={{
                  ...styles.pyramidModeChip,
                  ...(pyramidType === "up" ? styles.pyramidModeChipActive : {}),
                }}
                onClick={() => setPyramidType && setPyramidType("up")}
                title="Upscale only (optimized for small objects)"
              >
                Up Only
              </button>
              <button
                style={{
                  ...styles.pyramidModeChip,
                  ...(pyramidType === "down" ? styles.pyramidModeChipActive : {}),
                }}
                onClick={() => setPyramidType && setPyramidType("down")}
                title="Downscale only (optimized for large objects)"
              >
                Down Only
              </button>
            </div>

            {/* Scale badges indicator */}
            <div style={styles.scaleBadgesRow}>
              <span style={styles.scaleBadgesLabel}>Active Scales:</span>
              {(pyramidType === "both" || pyramidType === "down") && (
                <span style={styles.scaleTagDown}>0.71x Down</span>
              )}
              <span style={styles.scaleTagBase}>1.00x Base</span>
              {(pyramidType === "both" || pyramidType === "up") && (
                <>
                  <span style={styles.scaleTagUp}>1.41x Up</span>
                  <span style={styles.scaleTagUp}>2.00x Up</span>
                </>
              )}
            </div>
          </div>
        )}
      </div>

      {/* FPS Controller (Featured prominently for Stream / Video) */}
      <div
        style={{
          ...styles.card,
          ...(isStreamMode ? styles.cardHighlighted : {}),
        }}
      >
        <div style={styles.controlHeader}>
          <div style={styles.labelWithIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <polyline points="12 6 12 12 16 14" />
            </svg>
            <label style={styles.controlLabel}>Processing Rate (FPS)</label>
          </div>
          <span style={styles.badgeCyan}>{fps} FPS</span>
        </div>
        <p style={styles.controlHelp}>
          Controls frame sampling interval for model inference:{" "}
          <strong style={{ color: "#38bdf8" }}>~{Math.round(1000 / fps)}ms</strong> per frame
        </p>
        <div style={styles.sliderContainer}>
          <input
            id="fps-slider"
            type="range"
            min="1"
            max="30"
            step="1"
            value={fps}
            onChange={(e) => setFps(Number(e.target.value))}
            style={styles.slider}
          />
          <div style={styles.sliderTicks}>
            <span>1 FPS</span>
            <span>10 FPS</span>
            <span>20 FPS</span>
            <span>30 FPS</span>
          </div>
        </div>
      </div>

      {/* Confidence Threshold */}
      <div style={styles.card}>
        <div style={styles.controlHeader}>
          <div style={styles.labelWithIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#10b981" strokeWidth="2">
              <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
              <polyline points="22 4 12 14.01 9 11.01" />
            </svg>
            <label style={styles.controlLabel}>Confidence Cutoff</label>
          </div>
          <span style={styles.badgeEmerald}>{Math.round(confidence * 100)}%</span>
        </div>
        <p style={styles.controlHelp}>
          Filters out predictions with confidence below this threshold
        </p>
        <input
          id="conf-slider"
          type="range"
          min="0.05"
          max="0.95"
          step="0.01"
          value={confidence}
          onChange={(e) => setConfidence(Number(e.target.value))}
          style={styles.slider}
        />
        <div style={styles.sliderTicks}>
          <span>5%</span>
          <span>50%</span>
          <span>95%</span>
        </div>
      </div>

      {/* IoU Threshold */}
      <div style={styles.card}>
        <div style={styles.controlHeader}>
          <div style={styles.labelWithIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#8b5cf6" strokeWidth="2">
              <rect x="3" y="3" width="12" height="12" rx="2" />
              <rect x="9" y="9" width="12" height="12" rx="2" />
            </svg>
            <label style={styles.controlLabel}>IoU (NMS) Overlap</label>
          </div>
          <span style={styles.badgeViolet}>{Math.round(iouThreshold * 100)}%</span>
        </div>
        <p style={styles.controlHelp}>
          Non-Maximum Suppression threshold to merge overlapping boxes
        </p>
        <input
          id="iou-slider"
          type="range"
          min="0.1"
          max="0.9"
          step="0.05"
          value={iouThreshold}
          onChange={(e) => setIouThreshold(Number(e.target.value))}
          style={styles.slider}
        />
        <div style={styles.sliderTicks}>
          <span>0.10</span>
          <span>0.50</span>
          <span>0.90</span>
        </div>
      </div>

      {/* Class Filtering */}
      <div style={styles.card}>
        <div style={styles.controlHeader}>
          <div style={styles.labelWithIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2">
              <polygon points="12 2 2 7 12 12 22 7 12 2" />
              <polyline points="2 17 12 22 22 17" />
              <polyline points="2 12 12 17 22 12" />
            </svg>
            <label style={styles.controlLabel}>VOC 2007 Classes</label>
          </div>
          <span style={styles.badgeAmber}>
            {selectedClasses.length}/{allClasses.length}
          </span>
        </div>

        {/* Category shortcuts */}
        <div style={styles.categoryButtonGroup}>
          {Object.keys(VOC_CATEGORIES).map((cat) => (
            <button
              key={cat}
              style={styles.categoryChip}
              onClick={() => selectCategory(cat)}
            >
              {cat}
            </button>
          ))}
          <button
            style={{ ...styles.categoryChip, color: "#f43f5e" }}
            onClick={() => setSelectedClasses([])}
          >
            Clear
          </button>
        </div>

        {/* Scrollable class pills */}
        <div style={styles.classesGrid}>
          {allClasses.map((cls) => {
            const isSelected = selectedClasses.includes(cls);
            const color = classColors[cls] || "#38bdf8";
            return (
              <button
                key={cls}
                style={{
                  ...styles.classPill,
                  backgroundColor: isSelected
                    ? `${color}25`
                    : "rgba(255, 255, 255, 0.03)",
                  borderColor: isSelected ? color : "rgba(255, 255, 255, 0.08)",
                  color: isSelected ? "#f8fafc" : "#64748b",
                }}
                onClick={() => toggleClass(cls)}
              >
                <span
                  style={{
                    ...styles.classColorDot,
                    backgroundColor: color,
                    opacity: isSelected ? 1 : 0.3,
                  }}
                />
                <span>{cls}</span>
              </button>
            );
          })}
        </div>
      </div>

      {/* Visual Overlay Options */}
      <div style={styles.card}>
        <div style={styles.controlHeader}>
          <div style={styles.labelWithIcon}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#e2e8f0" strokeWidth="2">
              <path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z" />
              <circle cx="12" cy="12" r="3" />
            </svg>
            <label style={styles.controlLabel}>Display Toggles</label>
          </div>
        </div>

        <div style={styles.togglesList}>
          <label style={styles.toggleRow}>
            <span style={styles.toggleText}>Class Labels</span>
            <input
              type="checkbox"
              checked={showLabels}
              onChange={(e) => setShowLabels(e.target.checked)}
              style={styles.checkbox}
            />
          </label>

          <label style={styles.toggleRow}>
            <span style={styles.toggleText}>Confidence %</span>
            <input
              type="checkbox"
              checked={showConfidence}
              onChange={(e) => setShowConfidence(e.target.checked)}
              style={styles.checkbox}
            />
          </label>

          <label style={styles.toggleRow}>
            <span style={styles.toggleText}>Box Dimensions</span>
            <input
              type="checkbox"
              checked={showCoordinates}
              onChange={(e) => setShowCoordinates(e.target.checked)}
              style={styles.checkbox}
            />
          </label>

          <label style={styles.toggleRow}>
            <span style={styles.toggleText}>Anchor Grid Overlay</span>
            <input
              type="checkbox"
              checked={showGridGuide}
              onChange={(e) => setShowGridGuide(e.target.checked)}
              style={styles.checkbox}
            />
          </label>
        </div>
      </div>
    </aside>
  );
}

const styles = {
  sidebar: {
    width: "320px",
    flexShrink: 0,
    display: "flex",
    flexDirection: "column",
    gap: "14px",
    padding: "20px",
    background: "rgba(14, 19, 31, 0.6)",
    backdropFilter: "blur(16px)",
    borderRight: "1px solid rgba(255, 255, 255, 0.07)",
    overflowY: "auto",
    maxHeight: "calc(100vh - 72px)",
  },
  sectionHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    paddingBottom: "4px",
  },
  sectionTitle: {
    fontSize: "13px",
    fontWeight: "700",
    letterSpacing: "0.5px",
    textTransform: "uppercase",
    color: "#cbd5e1",
  },
  resetButton: {
    fontSize: "11px",
    color: "#38bdf8",
    background: "transparent",
    border: "none",
    cursor: "pointer",
    padding: "2px 6px",
    borderRadius: "4px",
  },
  card: {
    background: "rgba(18, 24, 38, 0.7)",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(255, 255, 255, 0.07)",
    borderRadius: "14px",
    padding: "14px",
    transition: "border-color 0.2s ease, box-shadow 0.2s ease",
  },
  cardHighlighted: {
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(56, 189, 248, 0.4)",
    boxShadow: "0 0 16px rgba(6, 182, 212, 0.15)",
  },
  controlHeader: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    marginBottom: "6px",
  },
  labelWithIcon: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  controlLabel: {
    fontSize: "13px",
    fontWeight: "600",
    color: "#f1f5f9",
  },
  controlHelp: {
    fontSize: "11px",
    color: "#94a3b8",
    marginBottom: "10px",
    lineHeight: "1.4",
  },
  badgeCyan: {
    fontSize: "11.5px",
    fontWeight: "700",
    color: "#38bdf8",
    background: "rgba(56, 189, 248, 0.12)",
    padding: "2px 8px",
    borderRadius: "6px",
    border: "1px solid rgba(56, 189, 248, 0.25)",
  },
  badgeEmerald: {
    fontSize: "11.5px",
    fontWeight: "700",
    color: "#10b981",
    background: "rgba(16, 185, 129, 0.12)",
    padding: "2px 8px",
    borderRadius: "6px",
    border: "1px solid rgba(16, 185, 129, 0.25)",
  },
  badgeViolet: {
    fontSize: "11.5px",
    fontWeight: "700",
    color: "#a855f7",
    background: "rgba(168, 85, 247, 0.12)",
    padding: "2px 8px",
    borderRadius: "6px",
    border: "1px solid rgba(168, 85, 247, 0.25)",
  },
  badgeAmber: {
    fontSize: "11.5px",
    fontWeight: "700",
    color: "#f59e0b",
    background: "rgba(245, 158, 11, 0.12)",
    padding: "2px 8px",
    borderRadius: "6px",
    border: "1px solid rgba(245, 158, 11, 0.25)",
  },
  sliderContainer: {
    width: "100%",
  },
  slider: {
    width: "100%",
    cursor: "pointer",
  },
  sliderTicks: {
    display: "flex",
    justifyContent: "space-between",
    fontSize: "10px",
    color: "#64748b",
    marginTop: "6px",
  },
  categoryButtonGroup: {
    display: "flex",
    flexWrap: "wrap",
    gap: "5px",
    marginBottom: "10px",
  },
  categoryChip: {
    fontSize: "10.5px",
    padding: "3px 8px",
    borderRadius: "6px",
    background: "rgba(255, 255, 255, 0.05)",
    border: "1px solid rgba(255, 255, 255, 0.08)",
    color: "#cbd5e1",
    cursor: "pointer",
    transition: "background-color 0.15s ease",
  },
  classesGrid: {
    display: "grid",
    gridTemplateColumns: "repeat(2, 1fr)",
    gap: "5px",
    maxHeight: "150px",
    overflowY: "auto",
    paddingRight: "4px",
  },
  classPill: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    padding: "5px 8px",
    borderRadius: "6px",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(255, 255, 255, 0.08)",
    fontSize: "11px",
    fontWeight: "500",
    cursor: "pointer",
    textAlign: "left",
    transition: "all 0.15s ease",
  },
  classColorDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    flexShrink: 0,
  },
  togglesList: {
    display: "flex",
    flexDirection: "column",
    gap: "8px",
    marginTop: "8px",
  },
  toggleRow: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    cursor: "pointer",
  },
  toggleText: {
    fontSize: "12px",
    color: "#cbd5e1",
  },
  checkbox: {
    accentColor: "#38bdf8",
    cursor: "pointer",
    width: "16px",
    height: "16px",
  },
  cardPyramidActive: {
    borderColor: "rgba(56, 189, 248, 0.35)",
    boxShadow: "0 0 16px rgba(56, 189, 248, 0.08)",
  },
  pyramidToggleBtn: {
    fontSize: "10.5px",
    fontWeight: "700",
    padding: "3px 10px",
    borderRadius: "12px",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(255, 255, 255, 0.1)",
    cursor: "pointer",
    letterSpacing: "0.5px",
    transition: "all 0.15s ease",
  },
  pyramidModeSelector: {
    display: "flex",
    gap: "5px",
    marginBottom: "8px",
  },
  pyramidModeChip: {
    flex: 1,
    fontSize: "10px",
    fontWeight: "600",
    padding: "4px 6px",
    borderRadius: "6px",
    background: "rgba(255, 255, 255, 0.04)",
    borderWidth: "1px",
    borderStyle: "solid",
    borderColor: "rgba(255, 255, 255, 0.08)",
    color: "#94a3b8",
    cursor: "pointer",
    textAlign: "center",
    transition: "all 0.15s ease",
  },
  pyramidModeChipActive: {
    background: "rgba(56, 189, 248, 0.16)",
    borderColor: "#38bdf8",
    color: "#38bdf8",
    boxShadow: "0 0 8px rgba(56, 189, 248, 0.2)",
  },
  scaleBadgesRow: {
    display: "flex",
    flexWrap: "wrap",
    alignItems: "center",
    gap: "5px",
    marginTop: "6px",
    paddingTop: "6px",
    borderTop: "1px solid rgba(255, 255, 255, 0.05)",
  },
  scaleBadgesLabel: {
    fontSize: "10px",
    color: "#64748b",
    fontWeight: "600",
  },
  scaleTagDown: {
    fontSize: "9.5px",
    fontWeight: "600",
    color: "#c084fc",
    background: "rgba(192, 132, 252, 0.12)",
    border: "1px solid rgba(192, 132, 252, 0.25)",
    padding: "2px 6px",
    borderRadius: "4px",
  },
  scaleTagBase: {
    fontSize: "9.5px",
    fontWeight: "600",
    color: "#38bdf8",
    background: "rgba(56, 189, 248, 0.12)",
    border: "1px solid rgba(56, 189, 248, 0.25)",
    padding: "2px 6px",
    borderRadius: "4px",
  },
  scaleTagUp: {
    fontSize: "9.5px",
    fontWeight: "600",
    color: "#34d399",
    background: "rgba(52, 211, 153, 0.12)",
    border: "1px solid rgba(52, 211, 153, 0.25)",
    padding: "2px 6px",
    borderRadius: "4px",
  },
};
