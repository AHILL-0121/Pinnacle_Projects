"""
Streamlit Frontend for Financial Document Analyzer
A clean, user-friendly interface for analyzing financial documents
"""
import streamlit as st
import requests
import io
import json
from typing import Optional
import time

# Page configuration
st.set_page_config(
    page_title="Financial Document Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A5F;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f8f9fa;
        border-radius: 10px;
        padding: 1rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1E3A5F;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1E3A5F;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #666;
    }
    .metric-change-positive {
        color: #28a745;
        font-size: 0.85rem;
    }
    .metric-change-negative {
        color: #dc3545;
        font-size: 0.85rem;
    }
    .insight-card {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        border-left: 4px solid #dc3545;
    }
    .risk-medium {
        border-left: 4px solid #ffc107;
    }
    .risk-low {
        border-left: 4px solid #28a745;
    }
    .section-header {
        font-size: 1.4rem;
        font-weight: 600;
        color: #1E3A5F;
        margin: 1.5rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e0e0e0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A5F;
    }
    .provider-card {
        background-color: #1a1a2e;
        border-radius: 8px;
        padding: 0.8rem;
        margin: 0.3rem 0;
    }
    .provider-available {
        border-left: 3px solid #28a745;
    }
    .provider-unavailable {
        border-left: 3px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

# Configuration
API_BASE_URL = "http://localhost:8000"

# Session state initialization
if 'analysis_result' not in st.session_state:
    st.session_state.analysis_result = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'providers' not in st.session_state:
    st.session_state.providers = None
if 'selected_provider' not in st.session_state:
    st.session_state.selected_provider = 'groq'


def fetch_providers():
    """Fetch available providers from backend"""
    try:
        response = requests.get(f"{API_BASE_URL}/providers", timeout=5)
        if response.status_code == 200:
            return response.json()
    except:
        pass
    return None


def format_value(value, unit=None):
    """Format numeric values for display"""
    if value is None:
        return "N/A"
    
    if not isinstance(value, (int, float)):
        return str(value)
    
    if unit == '%':
        return f"{value:.2f}%"
    elif unit == 'ratio':
        return f"{value:.2f}x"
    elif abs(value) >= 1_000_000_000:
        return f"${value/1_000_000_000:.2f}B"
    elif abs(value) >= 1_000_000:
        return f"${value/1_000_000:.2f}M"
    elif abs(value) >= 1_000:
        return f"${value/1_000:.2f}K"
    else:
        return f"${value:,.2f}"


def format_change(change, change_type=None):
    """Format change values"""
    if change is None:
        return ""
    
    sign = "+" if change > 0 else ""
    type_str = f" {change_type}" if change_type else ""
    
    if abs(change) < 1:
        return f"{sign}{change:.1%}{type_str}"
    else:
        return f"{sign}{change:.1f}%{type_str}"


def render_metric_card(name, value, unit=None, change=None, change_type=None):
    """Render a styled metric card"""
    formatted_value = format_value(value, unit)
    change_html = ""
    
    if change is not None:
        change_class = "metric-change-positive" if change > 0 else "metric-change-negative"
        arrow = "↑" if change > 0 else "↓"
        change_html = f'<span class="{change_class}">{arrow} {format_change(change, change_type)}</span>'
    
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{name.replace('_', ' ').title()}</div>
        <div class="metric-value">{formatted_value}</div>
        {change_html}
    </div>
    """, unsafe_allow_html=True)


def render_risk_card(risk):
    """Render a risk item card"""
    severity_class = f"risk-{risk.get('severity', 'medium')}"
    severity_badge = {
        'critical': '🔴 Critical',
        'high': '🟠 High',
        'medium': '🟡 Medium',
        'low': '🟢 Low'
    }.get(risk.get('severity', 'medium'), '🟡 Medium')
    
    st.markdown(f"""
    <div class="insight-card {severity_class}">
        <strong>{risk.get('category', 'Risk')}</strong> {severity_badge}<br/>
        <p>{risk.get('description', '')}</p>
        {f"<small><em>Mitigation: {risk.get('mitigation', '')}</em></small>" if risk.get('mitigation') else ""}
    </div>
    """, unsafe_allow_html=True)


def render_insight_card(insight):
    """Render an actionable insight card"""
    priority_badge = {
        'high': '🔥 High Priority',
        'medium': '⚡ Medium Priority',
        'low': '📌 Low Priority'
    }.get(insight.get('priority', 'medium'), '⚡ Medium')
    
    st.markdown(f"""
    <div class="insight-card">
        <strong>{insight.get('category', 'Insight')}</strong> {priority_badge}<br/>
        <p>{insight.get('insight', '')}</p>
        {f"<strong>Action:</strong> {insight.get('action', '')}" if insight.get('action') else ""}
    </div>
    """, unsafe_allow_html=True)


def analyze_document(file_bytes, filename, role, doc_type, company_name, fiscal_period, focus_areas, llm_provider=None):
    """Call the API to analyze the document with selected LLM provider"""
    try:
        files = {'file': (filename, file_bytes)}
        data = {
            'role': role,
        }
        
        if doc_type and doc_type != 'auto':
            data['document_type'] = doc_type
        if company_name:
            data['company_name'] = company_name
        if fiscal_period:
            data['fiscal_period'] = fiscal_period
        if focus_areas:
            data['focus_areas'] = focus_areas
        if llm_provider:
            data['llm_provider'] = llm_provider
        
        response = requests.post(
            f"{API_BASE_URL}/analyze",
            files=files,
            data=data,
            timeout=300
        )
        
        return response.json()
        
    except requests.exceptions.ConnectionError:
        return {
            'success': False,
            'message': 'Could not connect to the API server. Make sure it is running.'
        }
    except Exception as e:
        return {
            'success': False,
            'message': f'Error: {str(e)}'
        }


def main():
    # Header
    st.markdown('<p class="main-header">📊 Financial Document Analyzer</p>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">AI-powered analysis of financial reports with role-aware insights</p>', unsafe_allow_html=True)
    
    # Fetch providers status
    providers_data = fetch_providers()
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Analysis Settings")
        
        # LLM Provider Selection - NOW DYNAMIC PER REQUEST
        st.subheader("🤖 Select LLM Provider")
        st.caption("Choose the AI model for this analysis")
        
        if providers_data:
            providers = providers_data.get('providers', {})
            current_backend = providers_data.get('current_provider', 'groq')
            
            # Build provider options with availability status
            provider_options = []
            provider_labels = {}
            available_providers = []
            
            for key, info in providers.items():
                provider_options.append(key)
                status = "✅" if info['available'] else "❌"
                provider_labels[key] = f"{status} {info['name']} ({info['model']})"
                if info['available']:
                    available_providers.append(key)
            
            # Provider selection - SELECT FOR THIS ANALYSIS (not backend default)
            llm_provider = st.selectbox(
                "🎯 Select Provider for Analysis",
                options=provider_options,
                index=provider_options.index(current_backend) if current_backend in provider_options else 0,
                format_func=lambda x: provider_labels.get(x, x),
                help="Select which LLM to use for THIS analysis. You can change this for each document.",
                key="llm_selector"
            )
            
            # Check if selected provider is available
            selected_info = providers.get(llm_provider, {})
            if selected_info.get('available'):
                st.success(f"✅ **{llm_provider.upper()}** is ready")
            else:
                st.error(f"❌ **{llm_provider.upper()}** is not available")
                if llm_provider == 'ollama':
                    st.caption("💡 Start Ollama: `ollama serve`")
                elif llm_provider == 'groq':
                    st.caption("💡 Add GROQ_API_KEY to .env")
                elif llm_provider == 'gemini':
                    st.caption("💡 Add GEMINI_API_KEY to .env")
                
                # Suggest an available alternative
                if available_providers:
                    st.info(f"💡 Try: **{available_providers[0].upper()}** (available)")
            
            # Show all provider statuses in expander
            with st.expander("📊 All Provider Status", expanded=False):
                for key, info in providers.items():
                    status_icon = "🟢" if info['available'] else "🔴"
                    st.caption(f"{status_icon} {info['name']}: {info['model']}")
        else:
            # Fallback when API is not available
            st.error("❌ Cannot connect to backend")
            st.caption("Start backend: `cd backend && python run.py`")
            
            llm_provider = st.selectbox(
                "Select LLM Provider",
                options=['groq', 'gemini', 'ollama'],
                format_func=lambda x: {
                    'groq': '⚡ Groq (Fast, Free)',
                    'gemini': '🌟 Google Gemini',
                    'ollama': '🏠 Ollama (Local)'
                }.get(x, x),
                help="Choose which LLM to use"
            )
            
            # Show provider links
            if llm_provider == 'groq':
                st.caption("🔗 [Get free Groq API key](https://console.groq.com)")
            elif llm_provider == 'gemini':
                st.caption("🔗 [Get Gemini API key](https://makersuite.google.com)")
            elif llm_provider == 'ollama':
                st.caption("🔗 [Install Ollama](https://ollama.ai)")
        
        st.session_state.selected_provider = llm_provider
        
        st.divider()
        
        # Role selection
        role = st.selectbox(
            "👤 Select Your Role",
            options=['investor', 'analyst', 'auditor', 'executive'],
            format_func=lambda x: {
                'investor': '💰 Investor - Growth & Returns Focus',
                'analyst': '📈 Analyst - Ratios & Trends Focus',
                'auditor': '🔍 Auditor - Compliance & Red Flags',
                'executive': '👔 Executive - Strategic Overview'
            }.get(x, x),
            help="Your role determines the focus and tone of the analysis"
        )
        
        st.divider()
        
        # Document type
        doc_type = st.selectbox(
            "📄 Document Type",
            options=['auto', 'annual_report', 'quarterly_report', 'balance_sheet', 
                    'income_statement', 'cash_flow', 'investor_presentation'],
            format_func=lambda x: {
                'auto': '🔮 Auto-detect',
                'annual_report': '📅 Annual Report',
                'quarterly_report': '📊 Quarterly Report',
                'balance_sheet': '⚖️ Balance Sheet',
                'income_statement': '💵 Income Statement',
                'cash_flow': '💸 Cash Flow Statement',
                'investor_presentation': '🎯 Investor Presentation'
            }.get(x, x)
        )
        
        st.divider()
        
        # Optional fields
        st.subheader("📝 Optional Details")
        company_name = st.text_input("Company Name", placeholder="e.g., Apple Inc.")
        fiscal_period = st.text_input("Fiscal Period", placeholder="e.g., Q3 2024, FY2023")
        focus_areas = st.text_input(
            "Focus Areas", 
            placeholder="e.g., profitability, debt",
            help="Comma-separated areas to focus on"
        )
        
        st.divider()
        
        # Info
        st.info("""
        **Supported Formats:**
        - PDF (scanned or digital)
        - PNG, JPG, JPEG
        - TIFF, BMP
        
        **Max File Size:** 50 MB
        """)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown('<p class="section-header">📤 Upload Document</p>', unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader(
            "Choose a financial document",
            type=['pdf', 'png', 'jpg', 'jpeg', 'tiff', 'bmp', 'json'],
            help="Upload a scanned/digital financial document or JSON data file"
        )
        
        if uploaded_file:
            st.success(f"✅ **{uploaded_file.name}** ({uploaded_file.size / 1024:.1f} KB)")
            
            # Preview for images
            if uploaded_file.type.startswith('image'):
                st.image(uploaded_file, caption="Preview", use_container_width=True)
            
            # Preview for JSON
            if uploaded_file.name.endswith('.json'):
                try:
                    json_preview = json.loads(uploaded_file.getvalue().decode('utf-8'))
                    with st.expander("📄 JSON Preview", expanded=False):
                        st.json(json_preview)
                    uploaded_file.seek(0)  # Reset file pointer
                except json.JSONDecodeError:
                    st.warning("⚠️ Could not preview JSON - file may be malformed")
            
            # Analyze button
            if st.button("🚀 Analyze Document", type="primary", use_container_width=True):
                st.session_state.processing = True
                
                with st.spinner(f"🔄 Analyzing with {llm_provider.upper()}... This may take 30-60 seconds."):
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Simulate progress while waiting
                    steps = [
                        "📷 Preprocessing images...",
                        "📝 Extracting text via OCR...",
                        "📊 Detecting tables...",
                        "📈 Analyzing charts...",
                        "🔢 Extracting financial metrics...",
                        f"🤖 {llm_provider.upper()} analysis in progress...",
                        "📋 Generating summary..."
                    ]
                    
                    # Start analysis with selected provider
                    file_bytes = uploaded_file.getvalue()
                    result = analyze_document(
                        file_bytes,
                        uploaded_file.name,
                        role,
                        doc_type if doc_type != 'auto' else None,
                        company_name,
                        fiscal_period,
                        focus_areas,
                        llm_provider  # Pass the selected LLM provider
                    )
                    
                    progress_bar.progress(100)
                    status_text.text(f"✅ Analysis complete using {llm_provider.upper()}!")
                    
                    st.session_state.analysis_result = result
                    st.session_state.processing = False
    
    with col2:
        st.markdown('<p class="section-header">📋 Analysis Results</p>', unsafe_allow_html=True)
        
        if st.session_state.analysis_result:
            result = st.session_state.analysis_result
            
            if result.get('success'):
                data = result.get('data', {})
                
                # Confidence badge
                confidence = data.get('confidence_score', 0)
                conf_color = '#28a745' if confidence > 0.7 else '#ffc107' if confidence > 0.5 else '#dc3545'
                st.markdown(f"""
                <div style="text-align: right; margin-bottom: 1rem;">
                    <span style="background-color: {conf_color}; color: white; padding: 4px 12px; border-radius: 12px; font-size: 0.8rem;">
                        Confidence: {confidence:.0%}
                    </span>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.error(f"❌ {result.get('message', 'Analysis failed')}")
        else:
            st.info("👆 Upload a document and click 'Analyze' to see results here")
    
    # Results display
    if st.session_state.analysis_result and st.session_state.analysis_result.get('success'):
        data = st.session_state.analysis_result.get('data', {})
        
        st.divider()
        
        # Executive Summary
        st.markdown('<p class="section-header">📌 Executive Summary</p>', unsafe_allow_html=True)
        exec_summary = data.get('executive_summary') or {}
        
        if exec_summary.get('company_name'):
            st.subheader(exec_summary.get('company_name'))
        if exec_summary.get('period_covered'):
            st.caption(f"Period: {exec_summary.get('period_covered')}")
        
        st.write(exec_summary.get('overview', 'No summary available'))
        
        # Key highlights
        highlights = exec_summary.get('key_highlights', [])
        if highlights:
            st.markdown("**Key Highlights:**")
            for h in highlights:
                st.markdown(f"• {h}")
        
        st.divider()
        
        # Key Financial Metrics
        st.markdown('<p class="section-header">📊 Key Financial Metrics</p>', unsafe_allow_html=True)
        
        metrics = data.get('key_financial_metrics') or {}
        
        # Main metrics in columns
        col1, col2, col3, col4 = st.columns(4)
        
        metric_list = [
            ('revenue', 'Revenue'),
            ('net_profit', 'Net Profit'),
            ('ebitda', 'EBITDA'),
            ('eps', 'EPS')
        ]
        
        for col, (key, label) in zip([col1, col2, col3, col4], metric_list):
            with col:
                metric_data = metrics.get(key, {})
                if metric_data:
                    render_metric_card(
                        label,
                        metric_data.get('value'),
                        metric_data.get('unit'),
                        metric_data.get('change'),
                        metric_data.get('change_type')
                    )
                else:
                    render_metric_card(label, None)
        
        # Ratios
        st.markdown("**Financial Ratios:**")
        ratio_cols = st.columns(4)
        ratios = [
            ('roe', 'ROE', '%'),
            ('roa', 'ROA', '%'),
            ('current_ratio', 'Current Ratio', 'ratio'),
            ('debt_equity_ratio', 'Debt/Equity', 'ratio')
        ]
        
        for col, (key, label, unit) in zip(ratio_cols, ratios):
            with col:
                metric_data = metrics.get(key, {})
                if metric_data:
                    st.metric(
                        label, 
                        format_value(metric_data.get('value'), unit)
                    )
        
        st.divider()
        
        # Performance Analysis
        st.markdown('<p class="section-header">📈 Performance Analysis</p>', unsafe_allow_html=True)
        
        performance = data.get('performance_analysis') or {}
        
        perf_tabs = st.tabs(["Revenue", "Profitability", "Operations", "Comparisons"])
        
        with perf_tabs[0]:
            st.write(performance.get('revenue_analysis', 'No revenue analysis available'))
        
        with perf_tabs[1]:
            st.write(performance.get('profitability_analysis', 'No profitability analysis available'))
        
        with perf_tabs[2]:
            st.write(performance.get('operational_efficiency', 'No operational analysis available'))
        
        with perf_tabs[3]:
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("**Year-over-Year:**")
                st.write(performance.get('yoy_comparison', 'Not available'))
            with col2:
                st.markdown("**Quarter-over-Quarter:**")
                st.write(performance.get('qoq_comparison', 'Not available'))
        
        st.divider()
        
        # Chart Insights
        chart_insights = data.get('chart_insights') or []
        if chart_insights:
            st.markdown('<p class="section-header">📉 Chart & Graph Insights</p>', unsafe_allow_html=True)
            
            for i, chart in enumerate(chart_insights):
                with st.expander(f"📊 {chart.get('title', f'Chart {i+1}')} - {chart.get('chart_type', 'Unknown').title()}", expanded=i==0):
                    trend_icon = {'up': '📈', 'down': '📉', 'stable': '➡️', 'volatile': '📊'}.get(chart.get('trend'), '📊')
                    st.markdown(f"**Trend:** {trend_icon} {chart.get('trend', 'Unknown').title()}")
                    st.write(chart.get('insight', 'No insight available'))
                    
                    if chart.get('key_values'):
                        st.markdown("**Key Values:**")
                        for k, v in chart.get('key_values', {}).items():
                            st.markdown(f"• {k}: {v}")
        
        st.divider()
        
        # Risks and Outlook
        st.markdown('<p class="section-header">⚠️ Risks & Outlook</p>', unsafe_allow_html=True)
        
        risks = data.get('risks_and_outlook') or []
        if risks:
            for risk in risks:
                render_risk_card(risk)
        else:
            st.info("No significant risks identified")
        
        st.divider()
        
        # Actionable Insights
        st.markdown('<p class="section-header">💡 Actionable Insights</p>', unsafe_allow_html=True)
        
        insights = data.get('actionable_insights') or []
        if insights:
            for insight in insights:
                render_insight_card(insight)
        else:
            st.info("No specific actions recommended")
        
        st.divider()
        
        # Tables section
        tables = data.get('extracted_tables', [])
        if tables:
            st.markdown('<p class="section-header">📋 Extracted Tables</p>', unsafe_allow_html=True)
            
            for i, table in enumerate(tables):
                with st.expander(f"📊 {table.get('title', f'Table {i+1}')}", expanded=False):
                    if table.get('headers') and table.get('rows'):
                        import pandas as pd
                        df = pd.DataFrame(table['rows'], columns=table['headers'])
                        st.dataframe(df, use_container_width=True)
        
        # Export section
        st.divider()
        st.markdown('<p class="section-header">💾 Export Results</p>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            json_str = json.dumps(data, indent=2, default=str)
            st.download_button(
                "📥 Download JSON",
                json_str,
                file_name="financial_analysis.json",
                mime="application/json"
            )
        
        with col2:
            # Generate markdown report - use safe defaults for None values
            safe_metrics = metrics or {}
            safe_exec_summary = exec_summary or {}
            safe_performance = performance or {}
            safe_highlights = highlights or []
            safe_risks = risks or []
            safe_insights = insights or []
            
            report = f"""# Financial Analysis Report

## Executive Summary
{safe_exec_summary.get('overview', 'N/A')}

### Key Highlights
{chr(10).join(f'- {h}' for h in safe_highlights) if safe_highlights else 'N/A'}

## Key Metrics
- Revenue: {format_value(safe_metrics.get('revenue', {}).get('value') if safe_metrics.get('revenue') else None)}
- Net Profit: {format_value(safe_metrics.get('net_profit', {}).get('value') if safe_metrics.get('net_profit') else None)}
- ROE: {format_value(safe_metrics.get('roe', {}).get('value') if safe_metrics.get('roe') else None, '%')}

## Performance Analysis
{safe_performance.get('revenue_analysis', 'N/A')}

{safe_performance.get('profitability_analysis', 'N/A')}

## Risks
{chr(10).join(f'- [{r.get("severity", "medium").upper()}] {r.get("description", "")}' for r in safe_risks) if safe_risks else 'None identified'}

## Actionable Insights
{chr(10).join(f'- {i.get("insight", "")}' for i in safe_insights) if safe_insights else 'None'}

---
*Generated by Financial Document Analyzer*
"""
            st.download_button(
                "📄 Download Report",
                report,
                file_name="financial_report.md",
                mime="text/markdown"
            )
        
        # Processing info
        st.caption(f"⏱️ Processing time: {data.get('processing_time_seconds', 0):.1f}s | 📅 Generated: {data.get('generated_at', 'N/A')}")


if __name__ == "__main__":
    main()
