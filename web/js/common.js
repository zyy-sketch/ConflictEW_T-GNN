/* ConflictEW 共享工具库 v2（模块化重构）
 * 优化点（指令 D）：
 *  1. 模块化：CEW.api（API 层）/ CEW.data（数据获取，API 优先、静态兜底）/
 *     CEW.ui（ECharts 懒加载与初始化）/ CEW.util（工具函数）
 *  2. API 层：fetch 封装（超时/内存缓存/统一错误），服务器模式自动启用，
 *     服务器不可用时回退 window.APP_DATA 静态数据（file:// 兼容）
 *  3. ECharts 懒加载：IntersectionObserver，图表进入可视区才初始化
 *  4. 暗色主题：统一 CSS 变量（app.css），所有页面一致
 * 保持向后兼容：原有顶层函数（esc/D/riskLevel/initChart...）继续可用。
 */
(function () {
  "use strict";

  // ============================ 全局错误监听（立刻暴露问题） ============================
  // 任何 JS 异常 / Promise 拒绝都会在右上角显示一条红色错误条，
  // 不至于页面"完全空白"而无任何线索。
  window.addEventListener("error", function (e) {
    var msg = (e.error && (e.error.message || e.error.toString())) || e.message || "未知错误";
    console.error("[ConflictEW error]", msg, e.error && e.error.stack);
    showErrorBar(msg);
  });
  window.addEventListener("unhandledrejection", function (e) {
    var msg = (e.reason && (e.reason.message || e.reason.toString())) || "Promise 失败";
    console.error("[ConflictEW rejection]", msg);
    showErrorBar("异步请求失败：" + msg);
  });
  function showErrorBar(msg) {
    try {
      var el = document.getElementById("__cew_errbar");
      if (!el) {
        el = document.createElement("div");
        el.id = "__cew_errbar";
        el.style.cssText =
          "position:fixed;top:8px;right:8px;z-index:9999;max-width:520px;" +
          "padding:10px 14px;background:#5a1f1f;color:#ffd0d0;border:1px solid #e05a5a;" +
          "border-radius:8px;font:12px/1.5 Arial,Microsoft YaHei;box-shadow:0 4px 18px rgba(0,0,0,.4);";
        el.onclick = function () { el.remove(); };
        el.title = "点击关闭";
        document.body.appendChild(el);
      }
      var line = document.createElement("div");
      line.textContent = "⚠ " + msg;
      line.style.borderTop = "1px solid rgba(255,208,208,.2)";
      line.style.paddingTop = "6px";
      line.style.marginTop = "6px";
      el.appendChild(line);
    } catch (_) { /* 错误显示失败时静默 */ }
  }

  // ============================ 工具函数 ============================
  var util = {
    esc: function (s) {
      if (s === null || s === undefined) return "";
      return String(s)
        .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;").replace(/'/g, "&#39;");
    },
    fmtPct: function (v, d) { return (v === null || v === undefined || isNaN(v)) ? "—" : (v * 100).toFixed(d === undefined ? 1 : d) + "%"; },
    fmtNum: function (v, d) { return (v === null || v === undefined || isNaN(v)) ? "—" : Number(v).toFixed(d === undefined ? 3 : d); },
    fmtDate: function (s) { return s ? String(s).replace("T", " ").slice(0, 19) : "—"; },
    q: function (key, def) {
      var m = new URLSearchParams(location.search);
      var v = m.get(key);
      return v === null ? def : v;
    },
    deepMerge: function (dst) {
      for (var i = 1; i < arguments.length; i++) {
        var src = arguments[i];
        if (!src || typeof src !== "object") continue;
        for (var k in src) {
          if (Object.prototype.hasOwnProperty.call(src, k)) {
            var v = src[k];
            if (v && typeof v === "object" && !Array.isArray(v) && dst[k] && typeof dst[k] === "object") {
              util.deepMerge(dst[k], v);
            } else { dst[k] = v; }
          }
        }
      }
      return dst;
    }
  };

  // ============================ API 层 ============================
  var DEFAULT_T = { high: 0.7, mid: 0.5 };
  var api = {
    base: "api",            // 相对路径（服务器模式 /api）
    timeoutMs: 8000,
    _mem: {},               // 内存缓存
    _serverUp: null,        // null=未知, true/false 探测结果

    /** 探测服务器可用性（仅一次） */
    probe: function () {
      if (this._serverUp !== null) return Promise.resolve(this._serverUp);
      var self = this;
      return fetch(this.base + "/health", { method: "GET" })
        .then(function (r) { self._serverUp = r.ok; return r.ok; })
        .catch(function () { self._serverUp = false; return false; });
    },

    /** GET 请求：内存缓存 + 超时 */
    get: function (path, params) {
      var self = this;
      var qs = params ? "?" + new URLSearchParams(params).toString() : "";
      var url = this.base + "/" + path + qs;
      if (this._mem[url] !== undefined) return Promise.resolve(this._mem[url]);
      var ctrl = new AbortController();
      var timer = setTimeout(function () { ctrl.abort(); }, this.timeoutMs);
      return fetch(url, { signal: ctrl.signal })
        .then(function (r) {
          if (!r.ok) throw new Error("HTTP " + r.status);
          return r.json();
        })
        .then(function (body) {
          clearTimeout(timer);
          if (body && body.ok) {
            self._mem[url] = body.data;           // 只缓存 data 部分
            return body.data;
          }
          throw new Error((body && body.message) || "API 错误");
        })
        .catch(function (e) {
          clearTimeout(timer);
          self._serverUp = false;                 // 服务器不可达 → 后续走静态
          throw e;
        });
    },

    /** POST JSON */
    post: function (path, payload) {
      var self = this;
      return fetch(this.base + "/" + path, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
      }).then(function (r) { return r.json(); })
        .then(function (body) {
          if (body && body.ok) return body.data;
          throw new Error((body && body.message) || "API 错误");
        })
        .catch(function (e) { self._serverUp = false; throw e; });
    }
  };

  // ============================ 数据获取（API 优先 / 静态兜底） ============================
  // 各数据键 → [API 路径, 静态数据取值函数]
  var DATA_MAP = {
    "dashboard": ["dashboard", function (d) {
      var preds = d.predictions;
      var maxYear = Math.max.apply(null, preds.map(function (p) { return p.year; }));
      var latest = preds.filter(function (p) { return p.year === maxYear; });
      latest.sort(function (a, b) { return b.p_conflict - a.p_conflict; });
      var t = d.meta.thresholds;
      return {
        year: maxYear,
        kpi: {
          high_risk: latest.filter(function (p) { return p.p_conflict >= t.high; }).length,
          mid_risk: latest.filter(function (p) { return p.p_conflict >= t.mid && p.p_conflict < t.high; }).length,
          avg_risk: { value: latest.reduce(function (s, p) { return s + p.p_conflict; }, 0) / latest.length },
          alerts_count: d.alerts.length
        },
        top10: latest.slice(0, 10).map(function (p) {
          return { country: p.country, name_zh: p.name_zh, region: p.region, year: p.year, p_conflict: { value: p.p_conflict } };
        }),
        thresholds: t
      };
    }],
    "meta": ["meta", function (d) { return d.meta; }],
    "metrics": ["metrics", function (d) { return d.metrics; }],
    "alerts": ["alerts", function (d) { return d.alerts; }],
    "reports": ["reports", function (d) { return d.reports; }],
    "countries": ["countries", function (d) { return d.countries; }]
  };

  var data = {
    /** 获取数据：API 优先，静态兜底（同步场景直接返回静态数据 Promise） */
    get: function (key, params) {
      var map = DATA_MAP[key];
      if (!map) return Promise.reject(new Error("未知数据键: " + key));
      var self = this;
      return api.probe().then(function (up) {
        if (up) return api.get(map[0], params);
        return self.static(key);
      }).catch(function () { return self.static(key); });
    },
    /** 静态数据（window.APP_DATA，file:// 场景） */
    static: function (key) {
      var map = DATA_MAP[key];
      if (!map || !window.APP_DATA) return Promise.reject(new Error("无数据: " + key));
      return Promise.resolve(map[1](window.APP_DATA));
    },
    /** 是否有本地静态数据 */
    hasStatic: function () { return !!window.APP_DATA; }
  };

  // ============================ ECharts 懒加载 ============================
  var ui = {
    CHART_THEME: {
      textStyle: { color: "#93a1b8", fontFamily: "Microsoft YaHei" },
      color: ["#4d8dff", "#e05a5a", "#4bbf7a", "#e8934a", "#d9c04e", "#a78bfa", "#4dd0e1"],
      backgroundColor: "transparent",
      title: { textStyle: { color: "#e6ecf5", fontSize: 14 } },
      legend: { textStyle: { color: "#93a1b8" } },
      tooltip: { backgroundColor: "rgba(23,30,46,0.95)", borderColor: "#2b3752", textStyle: { color: "#e6ecf5", fontSize: 12.5 } },
      categoryAxis: { axisLine: { lineStyle: { color: "#2b3752" } }, axisLabel: { color: "#93a1b8" }, splitLine: { show: false }, axisTick: { lineStyle: { color: "#2b3752" } } },
      valueAxis: { axisLine: { lineStyle: { color: "#2b3752" } }, axisLabel: { color: "#93a1b8" }, splitLine: { lineStyle: { color: "rgba(43,55,82,0.5)" } } },
      grid: { top: 40, left: 60, right: 24, bottom: 40 }
    },

    initChart: function (el) {
      if (!window.echarts) { el.innerHTML = '<div class="empty">图表库未加载</div>'; return null; }
      return echarts.init(el, null, { renderer: "canvas" });
    },

    setChartOpt: function (chart, opt) {
      if (!chart) return;
      // 用闭包变量而非 this（严格模式下直接调用 this=undefined）
      chart.setOption(util.deepMerge({}, ui.CHART_THEME, opt), true);
    },

    resize: function (chart) { if (chart) setTimeout(function () { chart.resize(); }, 60); },

    /** 懒加载：图表进入可视区才初始化（指令 D：按需渲染） */
    lazyChart: function (el, initFn) {
      if (!el) return null;
      // 容器在视口外 → IntersectionObserver 延迟初始化
      if ("IntersectionObserver" in window) {
        var observed = false;
        var io = new IntersectionObserver(function (entries) {
          entries.forEach(function (en) {
            if (en.isIntersecting && !observed) {
              observed = true;
              initFn();
              io.disconnect();
            }
          });
        }, { rootMargin: "120px" });
        io.observe(el);
        return io;   // 返回 observer，调用方可 disconnect
      }
      initFn();       // 无 IO 支持 → 立即初始化
      return null;
    }
  };

  // ============================ 风险等级 / 国家（兼容原有） ============================
  var risk = {
    level: function (p) {
      var t = (window.APP_DATA && window.APP_DATA.meta && window.APP_DATA.meta.thresholds) || DEFAULT_T;
      if (p === null || p === undefined || isNaN(p)) return "na";
      if (p >= t.high) return "high";
      if (p >= t.mid) return "mid";
      return "low";
    },
    label: { high: "高风险", mid: "中风险", low: "低风险", na: "无数据" },
    color: { high: "#e05a5a", mid: "#e8934a", low: "#4bbf7a", na: "#5d6b85" },
    badge: function (lv) { return '<span class="badge ' + util.esc(lv) + '">' + util.esc(risk.label[lv] || lv) + "</span>"; },
    pBadge: function (p) { return risk.badge(risk.level(p)); }
  };

  var country = {
    byIso: function (iso3) {
      var d = window.APP_DATA;
      if (!d || !d.countryMap) return null;
      return d.countryMap[iso3] || null;
    },
    name: function (iso3) {
      var c = country.byIso(iso3);
      return c ? c.name_zh : iso3;
    },
    options: function (selected) {
      var d = window.APP_DATA;
      if (!d || !d.countries) return "";
      var opts = d.countries.slice().sort(function (a, b) { return a.name_zh.localeCompare(b.name_zh, "zh"); });
      return opts.map(function (c) {
        return '<option value="' + util.esc(c.iso3) + '"' + (c.iso3 === selected ? " selected" : "") + ">" +
          util.esc(c.name_zh) + "（" + util.esc(c.iso3) + "）</option>";
      }).join("");
    }
  };

  // ============================ 导航 / 状态 ============================
  var nav = {
    render: function (active) {
      var items = [
        ["index.html", "总览仪表盘", "P01"], ["data.html", "数据管理", "P02"],
        ["analysis.html", "预测分析", "P03"], ["visualize.html", "可视化中心", "P05"],
        ["alerts.html", "预警中心", "P06"], ["reports.html", "报告中心", "P07"]
      ];
      var el = document.getElementById("topnav");
      if (!el) return;
      var html = '<div class="brand">Conflict<em>EW</em></div>';
      items.forEach(function (it) {
        html += '<a class="nav' + (it[0] === active ? " active" : "") + '" href="' + it[0] + '">' + it[1] + "</a>";
      });
      html += '<div class="spacer"></div><div class="meta-info" id="nav-meta">' +
        '<span id="mode-tag" class="tag">静态</span></div>';
      el.innerHTML = html;
      var d = window.APP_DATA;
      var meta = document.getElementById("nav-meta");
      if (meta && d) {
        meta.innerHTML = '<span id="mode-tag" class="tag">静态</span> v' + (d.meta.version || "—") +
          " · 数据源 " + (d.meta.data_source || "—") + " · 种子 " + (d.meta.seed || "—");
      }
      // 探测服务器模式并在导航显示
      api.probe().then(function (up) {
        var tag = document.getElementById("mode-tag");
        if (tag) {
          tag.textContent = up ? "API" : "静态";
          tag.className = "tag " + (up ? "ok" : "");
        }
      });
    },
    noData: function (container, extra) {
      if (!container) return;
      container.innerHTML =
        '<div class="card" style="text-align:center;padding:60px 20px"><h3>暂无模型结果</h3>' +
        '<p style="color:var(--text-dim);margin:12px 0 20px">请先运行 <span class="mono">python scripts/run_all.py</span> 生成分析产物，' +
        '或启动服务 <span class="mono">python server.py</span>。</p>' +
        '<a class="btn" href="data.html">前往数据管理</a>' + (extra || "") + "</div>";
    }
  };

  // ============================ 导出（保持旧 API 兼容） ============================
  window.CEW = { api: api, data: data, ui: ui, util: util, risk: risk, country: country, nav: nav };

  // 旧接口别名（现有页面继续可用）
  window.esc = util.esc;
  window.fmtPct = util.fmtPct;
  window.fmtNum = util.fmtNum;
  window.fmtDate = util.fmtDate;
  window.q = util.q;
  window.D = function () { return window.APP_DATA || null; };
  window.hasData = function () { return !!window.APP_DATA && !!window.APP_DATA.meta; };
  window.riskLevel = risk.level;
  window.riskLabel = risk.label;
  window.riskColor = risk.color;
  window.badge = risk.badge;
  window.pBadge = risk.pBadge;
  window.CHART_THEME = ui.CHART_THEME;
  window.initChart = ui.initChart;
  window.setChartOpt = ui.setChartOpt;
  window.resizeChart = ui.resize;
  window.lazyChart = ui.lazyChart;
  window.registerWorldMap = function () {
    if (window.__mapReg) return true;
    if (!window.WORLD_GEO || !window.echarts) return false;
    var geo = window.WORLD_GEO;
    geo.features = (geo.features || []).map(function (f) {
      var name = (f.properties && f.properties.name) || "";
      if (name === "Taiwan") f.properties.name = "中国台湾";
      return f;
    });
    echarts.registerMap("world_cn", geo);
    window.__mapReg = true;
    return true;
  };
  window.countryByIso = country.byIso;
  window.countryName = country.name;
  window.countryOptions = country.options;
  window.renderNav = nav.render;
  window.renderNoData = nav.noData;
})();
