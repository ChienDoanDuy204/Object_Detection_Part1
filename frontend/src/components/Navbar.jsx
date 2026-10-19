"use client";

import React from "react";

export default function Navbar({
  activeTab,
  setActiveTab,
  backendInfo,
  backendStatus,
  streamStats,
}) {
  return (
    <header style={styles.header}>
      <div style={styles.headerLeft}>
        <div style={styles.logoBadge}>
          <svg width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="#38bdf8" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>
        </div>
        <div>
          <div style={styles.brandTitle}>
            <span>OverFeat</span>
            <span style={styles.brandSubtitle}>AI STUDIO</span>
          </div>
          <div style={styles.brandTagline}>
            Sliding-Window CNN Object Detection • VOC 2007
          </div>
        </div>
      </div>

      {/* Tabs */}
      <nav style={styles.navTabs}>
        <button
          id="tab-image"
          style={{
            ...styles.tabButton,
            ...(activeTab === "image" ? styles.tabButtonActive : {}),
          }}
          onClick={() => setActiveTab("image")}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="18" height="18" rx="2" ry="2"/>
            <circle cx="8.5" cy="8.5" r="1.5"/>
            <polyline points="21 15 16 10 5 21"/>
          </svg>
          <span>Image Analysis</span>
        </button>

        <button
          id="tab-video"
          style={{
            ...styles.tabButton,
            ...(activeTab === "video" ? styles.tabButtonActive : {}),
          }}
          onClick={() => setActiveTab("video")}
        >
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="23 7 16 12 23 17 23 7"/>
            <rect x="1" y="5" width="15" height="14" rx="2" ry="2"/>
          </svg>
          <span>Video Stream</span>
        </button>

        <button
          id="tab-camera"
          style={{
            ...styles.tabButton,
            ...(activeTab === "camera" ? styles.tabButtonActive : {}),
          }}
          onClick={() => setActiveTab("camera")}
        >
          <span style={styles.liveDot} />
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z"/>
            <circle cx="12" cy="13" r="4"/>
          </svg>
          <span>Camera Live Stream</span>
        </button>
      </nav>

      {/* Hardware / Backend Status Badge */}
      <div style={styles.headerRight}>
        <div style={styles.statusPill}>
          <span
            style={{
              ...styles.statusIndicator,
              backgroundColor:
                backendStatus === "connected"
                  ? "#10b981"
                  : backendStatus === "connecting"
                  ? "#f59e0b"
                  : "#ef4444",
              boxShadow:
                backendStatus === "connected"
                  ? "0 0 8px #10b981"
                  : backendStatus === "connecting"
                  ? "0 0 8px #f59e0b"
                  : "0 0 8px #ef4444",
            }}
          />
          <div style={styles.statusDetails}>
            <span style={styles.statusText}>
              {backendStatus === "connected"
                ? backendInfo?.device || "Backend Online"
                : backendStatus === "connecting"
                ? "Connecting..."
                : "Offline"}
            </span>
            {streamStats?.fps !== undefined && streamStats.fps > 0 && (
              <span style={styles.liveFpsBadge}>
                {streamStats.fps} FPS
              </span>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}

const styles = {
  header: {
    display: "flex",
    alignItems: "center",
    justifyContent: "space-between",
    padding: "14px 28px",
    background: "rgba(14, 19, 31, 0.8)",
    backdropFilter: "blur(16px)",
    borderBottom: "1px solid rgba(255, 255, 255, 0.08)",
    position: "sticky",
    top: 0,
    zIndex: 50,
  },
  headerLeft: {
    display: "flex",
    alignItems: "center",
    gap: "14px",
  },
  logoBadge: {
    width: "42px",
    height: "42px",
    borderRadius: "10px",
    background: "linear-gradient(135deg, rgba(6, 182, 212, 0.2), rgba(59, 130, 246, 0.1))",
    border: "1px solid rgba(56, 189, 248, 0.3)",
    display: "flex",
    alignItems: "center",
    justifyContent: "center",
    boxShadow: "0 0 20px rgba(6, 182, 212, 0.2)",
  },
  brandTitle: {
    fontSize: "19px",
    fontWeight: "700",
    letterSpacing: "-0.5px",
    color: "#f8fafc",
    display: "flex",
    alignItems: "baseline",
    gap: "8px",
  },
  brandSubtitle: {
    fontSize: "11px",
    fontWeight: "800",
    letterSpacing: "1.2px",
    background: "linear-gradient(90deg, #38bdf8, #818cf8)",
    WebkitBackgroundClip: "text",
    WebkitTextFillColor: "transparent",
  },
  brandTagline: {
    fontSize: "11.5px",
    color: "#94a3b8",
  },
  navTabs: {
    display: "flex",
    alignItems: "center",
    gap: "6px",
    background: "rgba(8, 11, 17, 0.6)",
    padding: "4px",
    borderRadius: "12px",
    border: "1px solid rgba(255, 255, 255, 0.06)",
  },
  tabButton: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
    padding: "8px 18px",
    fontSize: "13.5px",
    fontWeight: "500",
    color: "#94a3b8",
    background: "transparent",
    border: "none",
    borderRadius: "9px",
    cursor: "pointer",
    transition: "all 0.2s ease",
  },
  tabButtonActive: {
    color: "#ffffff",
    background: "linear-gradient(135deg, rgba(6, 182, 212, 0.25), rgba(59, 130, 246, 0.2))",
    border: "1px solid rgba(56, 189, 248, 0.4)",
    boxShadow: "0 2px 10px rgba(6, 182, 212, 0.2)",
  },
  liveDot: {
    width: "7px",
    height: "7px",
    borderRadius: "50%",
    backgroundColor: "#ef4444",
    boxShadow: "0 0 6px #ef4444",
  },
  headerRight: {
    display: "flex",
    alignItems: "center",
    gap: "12px",
  },
  statusPill: {
    display: "flex",
    alignItems: "center",
    gap: "10px",
    padding: "6px 14px",
    background: "rgba(18, 24, 38, 0.8)",
    borderRadius: "20px",
    border: "1px solid rgba(255, 255, 255, 0.07)",
  },
  statusIndicator: {
    width: "8px",
    height: "8px",
    borderRadius: "50%",
    transition: "background-color 0.3s ease",
  },
  statusDetails: {
    display: "flex",
    alignItems: "center",
    gap: "8px",
  },
  statusText: {
    fontSize: "12px",
    fontWeight: "500",
    color: "#cbd5e1",
  },
  liveFpsBadge: {
    fontSize: "10.5px",
    fontWeight: "600",
    color: "#10b981",
    background: "rgba(16, 185, 129, 0.15)",
    padding: "2px 6px",
    borderRadius: "6px",
    border: "1px solid rgba(16, 185, 129, 0.3)",
  },
};
