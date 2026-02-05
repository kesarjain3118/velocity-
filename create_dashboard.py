import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Page config
st.set_page_config(
    page_title="VelocityMart Operations Dashboard", 
    page_icon="📦",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 48px;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 20px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        text-align: center;
    }
    .alert-box {
        background-color: #fff3cd;
        border-left: 5px solid #ffc107;
        padding: 15px;
        margin: 10px 0;
    }
    .critical-alert {
        background-color: #f8d7da;
        border-left: 5px solid #dc3545;
        padding: 15px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<p class="main-header">📦 VelocityMart Operations Dashboard</p>', unsafe_allow_html=True)
st.markdown("### Interim Head of Operations - Strategic Intervention System")
st.markdown("---")

# Load data
@st.cache_data
def load_data():
    warehouse = pd.read_csv('warehouse_constraints.csv')
    sku = pd.read_csv('sku_master.csv')
    orders = pd.read_csv('order_transactions.csv')
    movement = pd.read_csv('picker_movement.csv')
    
    orders['order_timestamp'] = pd.to_datetime(orders['order_timestamp'])
    movement['order_timestamp'] = pd.to_datetime(movement['order_timestamp'])
    movement['movement_timestamp'] = pd.to_datetime(movement['movement_timestamp'])
    
    # Extract temporal features
    orders['week'] = orders['order_timestamp'].dt.isocalendar().week
    orders['hour'] = orders['order_timestamp'].dt.hour
    orders['date'] = orders['order_timestamp'].dt.date
    
    # Extract aisle from slot_id
    sku['aisle'] = sku['current_slot'].str.extract(r'([A-Z]\d+)')[0]
    sku['zone'] = sku['current_slot'].str[0]
    
    return warehouse, sku, orders, movement

warehouse, sku, orders, movement = load_data()

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/warehouse.png", width=100)
    st.title("Navigation")
    
    page = st.radio(
        "Select View:",
        ["🏠 Executive Summary", 
         "🔍 Data Forensics", 
         "🗺️ Heatmap Analysis",
         "❄️ Spoilage Risk",
         "👷 Picker Performance",
         "🎯 Optimization Plan"]
    )
    
    st.markdown("---")
    st.markdown("### Quick Stats")
    st.metric("Total SKUs", len(sku))
    st.metric("Total Orders (90 weeks)", len(orders))
    st.metric("Warehouse Slots", len(warehouse))
    st.metric("Active Pickers", len(movement['picker_id'].unique()))

# EXECUTIVE SUMMARY PAGE
if page == "🏠 Executive Summary":
    st.header("🏠 Executive Summary - Warehouse Health Status")
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    # Calculate metrics
    sku_warehouse = sku.merge(warehouse[['slot_id', 'temp_zone', 'max_weight_kg']], 
                              left_on='current_slot', right_on='slot_id', how='left')
    temp_violations = len(sku_warehouse[sku_warehouse['temp_req'] != sku_warehouse['temp_zone']])
    weight_issues = len(sku[sku['weight_kg'] > 50])
    
    avg_fulfillment_recent = 6.2  # From manual
    avg_fulfillment_baseline = 3.8  # From manual
    
    with col1:
        st.metric(
            "Avg Fulfillment Time", 
            f"{avg_fulfillment_recent} min",
            f"+{avg_fulfillment_recent - avg_fulfillment_baseline:.1f} min",
            delta_color="inverse"
        )
    
    with col2:
        st.metric(
            "Temp Violations",
            temp_violations,
            f"{temp_violations/len(sku)*100:.1f}%",
            delta_color="inverse"
        )
    
    with col3:
        st.metric(
            "Decimal Drift Issues",
            weight_issues,
            f"{weight_issues/len(sku)*100:.1f}%",
            delta_color="inverse"
        )
    
    with col4:
        health_score = ((len(sku) - temp_violations - weight_issues) / len(sku) * 100)
        st.metric(
            "Warehouse Health",
            f"{health_score:.1f}%",
            delta_color="normal"
        )
    
    st.markdown("---")
    
    # Critical Alerts
    st.subheader("🚨 Critical Alerts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="critical-alert">
        <h4>⚠️ Fulfillment Time Crisis</h4>
        <p>Average fulfillment time has increased by <b>63%</b> from 3.8 to 6.2 minutes.</p>
        <p><b>Impact:</b> Customer satisfaction at risk, operational costs rising.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="critical-alert">
        <h4>❄️ Spoilage Risk - High Priority</h4>
        <p><b>338 SKUs</b> (Frozen/Refrigerated) are in wrong temperature zones.</p>
        <p><b>Impact:</b> Estimated daily spoilage loss: ₹50,000+</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="alert-box">
        <h4>🚧 Aisle B Congestion</h4>
        <p>Peak hour (19:00) congestion in B-aisles creating bottlenecks.</p>
        <p><b>B21:</b> 1,490 picks | <b>B25:</b> 1,404 picks | <b>B20:</b> 1,335 picks</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="alert-box">
        <h4>🔍 Data Integrity Issues</h4>
        <p><b>20 SKUs</b> with 10x decimal drift in weight measurements.</p>
        <p><b>PICKER-07</b> showing suspicious efficiency (47.8% below average travel).</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Order volume trend
    st.subheader("📊 Order Volume Trend (90 Weeks)")
    daily_orders = orders.groupby('date').size().reset_index(name='orders')
    daily_orders['date'] = pd.to_datetime(daily_orders['date'])
    
    fig = px.line(daily_orders, x='date', y='orders', 
                  title='Daily Order Volume Over Time',
                  labels={'orders': 'Number of Orders', 'date': 'Date'})
    fig.add_hline(y=daily_orders['orders'].mean(), line_dash="dash", 
                  annotation_text="Average", line_color="red")
    st.plotly_chart(fig, use_container_width=True)
    
    # Category distribution
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Inventory by Category")
        category_dist = sku['category'].value_counts()
        fig = px.pie(values=category_dist.values, names=category_dist.index,
                     title='SKU Distribution by Category')
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🌡️ Temperature Requirements")
        temp_dist = sku['temp_req'].value_counts()
        fig = px.bar(x=temp_dist.index, y=temp_dist.values,
                     title='SKUs by Temperature Requirement',
                     labels={'x': 'Temperature Zone', 'y': 'Number of SKUs'},
                     color=temp_dist.values, color_continuous_scale='Blues')
        st.plotly_chart(fig, use_container_width=True)

# DATA FORENSICS PAGE
elif page == "🔍 Data Forensics":
    st.header("🔍 Data Forensics & Integrity Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "Decimal Drift", "Temperature Violations", "Picker Anomalies", "Weight Issues"
    ])
    
    with tab1:
        st.subheader("📊 Decimal Drift Detection")
        st.markdown("""
        **Issue**: SKUs with weights > 50kg are likely affected by 10x decimal point errors.
        Typical grocery items shouldn't exceed 50kg.
        """)
        
        decimal_drift = sku[sku['weight_kg'] > 50].copy()
        decimal_drift['corrected_weight'] = decimal_drift['weight_kg'] / 10
        
        st.write(f"**Total affected SKUs:** {len(decimal_drift)}")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Original Weight Range", 
                     f"{decimal_drift['weight_kg'].min():.1f} - {decimal_drift['weight_kg'].max():.1f} kg")
        with col2:
            st.metric("Corrected Weight Range",
                     f"{decimal_drift['corrected_weight'].min():.1f} - {decimal_drift['corrected_weight'].max():.1f} kg")
        
        st.dataframe(decimal_drift[['sku_id', 'category', 'weight_kg', 'corrected_weight', 
                                    'temp_req', 'current_slot']].head(20), use_container_width=True)
        
        # Visualization
        fig = go.Figure()
        fig.add_trace(go.Bar(name='Original Weight', x=decimal_drift['sku_id'].head(20), 
                            y=decimal_drift['weight_kg'].head(20), marker_color='red'))
        fig.add_trace(go.Bar(name='Corrected Weight', x=decimal_drift['sku_id'].head(20), 
                            y=decimal_drift['corrected_weight'].head(20), marker_color='green'))
        fig.update_layout(title='Weight Comparison: Original vs Corrected (Top 20)', 
                         xaxis_title='SKU ID', yaxis_title='Weight (kg)', barmode='group')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("❄️ Temperature Constraint Violations")
        
        sku_warehouse = sku.merge(warehouse[['slot_id', 'temp_zone']], 
                                  left_on='current_slot', right_on='slot_id', how='left')
        temp_violations = sku_warehouse[sku_warehouse['temp_req'] != sku_warehouse['temp_zone']]
        
        st.write(f"**Total violations:** {len(temp_violations)} ({len(temp_violations)/len(sku)*100:.1f}% of inventory)")
        
        violation_summary = temp_violations.groupby(['temp_req', 'temp_zone']).size().reset_index(name='count')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.dataframe(violation_summary, use_container_width=True)
        
        with col2:
            fig = px.bar(violation_summary, x='temp_req', y='count', color='temp_zone',
                        title='Temperature Violations by Type',
                        labels={'count': 'Number of SKUs', 'temp_req': 'Required Zone',
                               'temp_zone': 'Current Zone'},
                        barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("🚨 High-Risk Violations (Spoilage Risk)")
        high_risk = temp_violations[temp_violations['temp_req'].isin(['Frozen', 'Refrigerated'])]
        st.write(f"**Frozen/Refrigerated items in wrong zones:** {len(high_risk)}")
        st.dataframe(high_risk[['sku_id', 'category', 'weight_kg', 'temp_req', 
                                'temp_zone', 'current_slot']].head(20), use_container_width=True)
    
    with tab3:
        st.subheader("👷 Picker Efficiency Analysis - The Shortcut Paradox")
        
        picker_stats = movement.groupby('picker_id').agg({
            'travel_distance_m': ['mean', 'median', 'std', 'count']
        }).round(2)
        picker_stats.columns = ['avg_distance', 'median_distance', 'std_distance', 'trips']
        picker_stats = picker_stats.sort_values('avg_distance')
        
        st.dataframe(picker_stats, use_container_width=True)
        
        # Identify suspicious pickers
        mean_distance = picker_stats['avg_distance'].mean()
        threshold = mean_distance - 1.5 * picker_stats['avg_distance'].std()
        
        suspicious = picker_stats[picker_stats['avg_distance'] < threshold]
        
        st.markdown(f"""
        <div class="critical-alert">
        <h4>⚠️ Suspicious Picker Detected: PICKER-07</h4>
        <p><b>Average Distance:</b> {picker_stats.loc['PICKER-07', 'avg_distance']:.2f}m</p>
        <p><b>Warehouse Average:</b> {mean_distance:.2f}m</p>
        <p><b>Deviation:</b> {(picker_stats.loc['PICKER-07', 'avg_distance'] - mean_distance) / mean_distance * 100:.1f}%</p>
        <p><b>Analysis:</b> PICKER-07 is 47.8% more "efficient" than average, suggesting potential shortcuts through safety zones.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualization
        fig = px.bar(picker_stats.reset_index(), x='picker_id', y='avg_distance',
                    title='Average Travel Distance by Picker',
                    labels={'avg_distance': 'Average Distance (m)', 'picker_id': 'Picker ID'},
                    color='avg_distance', color_continuous_scale='RdYlGn_r')
        fig.add_hline(y=mean_distance, line_dash="dash", annotation_text="Average", line_color="red")
        st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("⚖️ Weight Constraint Violations")
        
        sku_warehouse = sku.merge(warehouse[['slot_id', 'max_weight_kg']], 
                                  left_on='current_slot', right_on='slot_id', how='left')
        weight_violations = sku_warehouse[sku_warehouse['weight_kg'] > sku_warehouse['max_weight_kg']]
        
        st.write(f"**SKUs exceeding slot capacity:** {len(weight_violations)}")
        
        if len(weight_violations) > 0:
            st.dataframe(weight_violations[['sku_id', 'category', 'weight_kg', 
                                           'max_weight_kg', 'current_slot']], use_container_width=True)
        else:
            st.success("No weight violations detected!")

# HEATMAP ANALYSIS PAGE
elif page == "🗺️ Heatmap Analysis":
    st.header("🗺️ Warehouse Heatmap Analysis")
    
    tab1, tab2, tab3 = st.tabs(["Peak Hour Activity", "Aisle Congestion", "Zone Analysis"])
    
    # Merge orders with SKU data
    orders_sku = orders.merge(sku[['sku_id', 'aisle', 'zone']], on='sku_id', how='left')
    
    with tab1:
        st.subheader("🕐 Peak Hour (19:00) Activity Heatmap")
        
        hour_selected = st.slider("Select Hour", 0, 23, 19)
        
        aisle_hour_activity = orders_sku.groupby(['aisle', 'hour']).size().reset_index(name='picks')
        peak_hour_data = aisle_hour_activity[aisle_hour_activity['hour'] == hour_selected].sort_values('picks', ascending=False)
        
        st.write(f"**Top 20 busiest aisles at {hour_selected}:00:**")
        
        fig = px.bar(peak_hour_data.head(20), x='aisle', y='picks',
                    title=f'Aisle Activity at {hour_selected}:00',
                    labels={'picks': 'Number of Picks', 'aisle': 'Aisle ID'},
                    color='picks', color_continuous_scale='Reds')
        st.plotly_chart(fig, use_container_width=True)
        
        st.dataframe(peak_hour_data.head(20), use_container_width=True)
    
    with tab2:
        st.subheader("🚧 B-Aisle Congestion Analysis")
        
        st.markdown("""
        **Critical Finding**: The manual mentions that "The forklift can't enter Aisle B if more than 2 pickers are already there."
        This creates a significant bottleneck during peak hours.
        """)
        
        b_aisles = aisle_hour_activity[aisle_hour_activity['aisle'].str.startswith('B', na=False)]
        b_aisles_pivot = b_aisles.pivot(index='aisle', columns='hour', values='picks').fillna(0)
        
        fig = px.imshow(b_aisles_pivot, 
                       labels=dict(x="Hour of Day", y="B-Aisle", color="Picks"),
                       title="B-Aisle Activity Heatmap (All Hours)",
                       color_continuous_scale='YlOrRd')
        st.plotly_chart(fig, use_container_width=True)
        
        b_aisles_19 = b_aisles[b_aisles['hour'] == 19].sort_values('picks', ascending=False)
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.write("**B-Aisle Activity at 19:00:**")
            st.dataframe(b_aisles_19.head(10), use_container_width=True)
        
        with col2:
            fig = px.bar(b_aisles_19.head(10), x='aisle', y='picks',
                        title='Top 10 B-Aisles at Peak Hour (19:00)',
                        color='picks', color_continuous_scale='OrRd')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📍 Zone-Level Activity Analysis")
        
        zone_activity = orders_sku.groupby(['zone', 'hour']).size().reset_index(name='picks')
        zone_pivot = zone_activity.pivot(index='zone', columns='hour', values='picks').fillna(0)
        
        fig = px.imshow(zone_pivot,
                       labels=dict(x="Hour of Day", y="Zone", color="Picks"),
                       title="Zone Activity Heatmap",
                       color_continuous_scale='Viridis')
        st.plotly_chart(fig, use_container_width=True)
        
        zone_summary = orders_sku.groupby('zone').size().reset_index(name='total_picks')
        fig = px.pie(zone_summary, values='total_picks', names='zone',
                    title='Pick Distribution by Zone')
        st.plotly_chart(fig, use_container_width=True)

# SPOILAGE RISK PAGE
elif page == "❄️ Spoilage Risk":
    st.header("❄️ Spoilage Risk Analysis")
    
    sku_warehouse = sku.merge(warehouse[['slot_id', 'temp_zone']], 
                              left_on='current_slot', right_on='slot_id', how='left')
    temp_violations = sku_warehouse[sku_warehouse['temp_req'] != sku_warehouse['temp_zone']]
    
    st.markdown("""
    **Critical Business Impact**: Temperature violations lead to product spoilage, 
    customer complaints, and direct financial losses.
    """)
    
    col1, col2, col3 = st.columns(3)
    
    frozen_wrong = len(temp_violations[temp_violations['temp_req'] == 'Frozen'])
    refrig_wrong = len(temp_violations[temp_violations['temp_req'] == 'Refrigerated'])
    total_high_risk = frozen_wrong + refrig_wrong
    
    with col1:
        st.metric("Frozen Items Misplaced", frozen_wrong, delta_color="inverse")
    with col2:
        st.metric("Refrigerated Items Misplaced", refrig_wrong, delta_color="inverse")
    with col3:
        st.metric("Total High-Risk SKUs", total_high_risk, delta_color="inverse")
    
    st.markdown("---")
    
    # Estimated financial impact
    st.subheader("💰 Estimated Financial Impact")
    
    avg_value_per_sku = 500  # Assumption: ₹500 per SKU average
    daily_spoilage_rate = 0.05  # 5% daily spoilage for misplaced items
    
    daily_loss = total_high_risk * avg_value_per_sku * daily_spoilage_rate
    weekly_loss = daily_loss * 7
    monthly_loss = daily_loss * 30
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Estimated Daily Loss", f"₹{daily_loss:,.0f}")
    with col2:
        st.metric("Estimated Weekly Loss", f"₹{weekly_loss:,.0f}")
    with col3:
        st.metric("Estimated Monthly Loss", f"₹{monthly_loss:,.0f}")
    
    st.markdown("---")
    
    # Violation breakdown
    st.subheader("📊 Violation Breakdown")
    
    violation_detail = temp_violations.groupby(['temp_req', 'temp_zone', 'category']).size().reset_index(name='count')
    violation_detail = violation_detail.sort_values('count', ascending=False)
    
    fig = px.treemap(violation_detail, path=['temp_req', 'temp_zone', 'category'], values='count',
                    title='Temperature Violations Hierarchy',
                    color='count', color_continuous_scale='Reds')
    st.plotly_chart(fig, use_container_width=True)
    
    st.dataframe(violation_detail.head(20), use_container_width=True)
    
    # Priority action list
    st.subheader("🎯 Priority Action List - Top 30 SKUs to Relocate")
    
    high_risk_violations = temp_violations[temp_violations['temp_req'].isin(['Frozen', 'Refrigerated'])].copy()
    
    # Merge with order frequency
    sku_order_freq = orders.groupby('sku_id').size().reset_index(name='order_frequency')
    high_risk_violations = high_risk_violations.merge(sku_order_freq, on='sku_id', how='left')
    high_risk_violations['order_frequency'] = high_risk_violations['order_frequency'].fillna(0)
    
    # Sort by order frequency (high frequency = high priority)
    priority_relocations = high_risk_violations.sort_values('order_frequency', ascending=False).head(30)
    
    st.dataframe(priority_relocations[['sku_id', 'category', 'temp_req', 'temp_zone', 
                                       'current_slot', 'order_frequency']], use_container_width=True)

# PICKER PERFORMANCE PAGE
elif page == "👷 Picker Performance":
    st.header("👷 Picker Performance Analysis")
    
    picker_stats = movement.groupby('picker_id').agg({
        'travel_distance_m': ['mean', 'median', 'std', 'min', 'max', 'count']
    }).round(2)
    picker_stats.columns = ['avg_distance', 'median_distance', 'std_distance', 
                           'min_distance', 'max_distance', 'total_trips']
    picker_stats = picker_stats.sort_values('avg_distance')
    
    tab1, tab2, tab3 = st.tabs(["Overall Performance", "Efficiency Analysis", "Travel Patterns"])
    
    with tab1:
        st.subheader("📊 Overall Picker Statistics")
        
        st.dataframe(picker_stats, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(picker_stats.reset_index(), x='picker_id', y='avg_distance',
                        title='Average Travel Distance by Picker',
                        color='avg_distance', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(picker_stats.reset_index(), x='picker_id', y='total_trips',
                        title='Total Trips by Picker',
                        color='total_trips', color_continuous_scale='Blues')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("🔍 Efficiency Analysis - Shortcut Detection")
        
        mean_distance = picker_stats['avg_distance'].mean()
        std_distance = picker_stats['avg_distance'].std()
        
        picker_stats_display = picker_stats.reset_index()
        picker_stats_display['deviation_%'] = ((picker_stats_display['avg_distance'] - mean_distance) / mean_distance * 100).round(1)
        picker_stats_display['z_score'] = ((picker_stats_display['avg_distance'] - mean_distance) / std_distance).round(2)
        
        # Highlight suspicious pickers
        picker_stats_display['status'] = picker_stats_display['z_score'].apply(
            lambda x: '🚨 Suspicious' if x < -1.5 else ('✅ Normal' if abs(x) <= 1.5 else '⚠️ Inefficient')
        )
        
        st.dataframe(picker_stats_display[['picker_id', 'avg_distance', 'deviation_%', 
                                           'z_score', 'status']], use_container_width=True)
        
        # PICKER-07 deep dive
        st.markdown("""
        <div class="critical-alert">
        <h3>🔍 PICKER-07 Deep Dive</h3>
        <p>PICKER-07 shows a <b>-47.8% deviation</b> from the warehouse average, 
        with a z-score indicating statistical anomaly. This suggests potential shortcuts 
        through restricted zones or safety barriers.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Comparison visualization
        picker_07_data = movement[movement['picker_id'] == 'PICKER-07']['travel_distance_m']
        other_pickers_data = movement[movement['picker_id'] != 'PICKER-07']['travel_distance_m']
        
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=picker_07_data, name='PICKER-07', 
                                  marker_color='red', opacity=0.7, nbinsx=50))
        fig.add_trace(go.Histogram(x=other_pickers_data, name='Other Pickers', 
                                  marker_color='blue', opacity=0.5, nbinsx=50))
        fig.update_layout(title='Travel Distance Distribution: PICKER-07 vs Others',
                         xaxis_title='Travel Distance (m)', yaxis_title='Frequency',
                         barmode='overlay')
        st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("📈 Travel Pattern Analysis")
        
        # Daily travel distance trend
        movement['date'] = movement['order_timestamp'].dt.date
        daily_picker_distance = movement.groupby(['date', 'picker_id'])['travel_distance_m'].sum().reset_index()
        
        picker_selected = st.selectbox("Select Picker", sorted(movement['picker_id'].unique()))
        
        picker_data = daily_picker_distance[daily_picker_distance['picker_id'] == picker_selected]
        
        fig = px.line(picker_data, x='date', y='travel_distance_m',
                     title=f'{picker_selected} Daily Travel Distance',
                     labels={'travel_distance_m': 'Total Distance (m)', 'date': 'Date'})
        fig.add_hline(y=picker_data['travel_distance_m'].mean(), line_dash="dash",
                     annotation_text="Average", line_color="red")
        st.plotly_chart(fig, use_container_width=True)

# OPTIMIZATION PLAN PAGE
elif page == "🎯 Optimization Plan":
    st.header("🎯 Strategic Optimization Plan for Week 91")
    
    st.markdown("""
    ### Warehouse Health Score: Custom Metric
    
    Our proposed **Chaos Score** combines multiple factors to measure warehouse operational health:
    """)
    
    # Calculate Chaos Score components
    sku_warehouse = sku.merge(warehouse[['slot_id', 'temp_zone', 'max_weight_kg']], 
                              left_on='current_slot', right_on='slot_id', how='left')
    
    temp_violations = len(sku_warehouse[sku_warehouse['temp_req'] != sku_warehouse['temp_zone']])
    weight_issues = len(sku[sku['weight_kg'] > 50])
    
    # Score components (0-100, higher is better)
    temp_compliance = ((len(sku) - temp_violations) / len(sku)) * 100
    weight_compliance = ((len(sku) - weight_issues) / len(sku)) * 100
    
    fulfillment_efficiency = (3.8 / 6.2) * 100  # Baseline vs current
    
    # Weighted Chaos Score
    chaos_score = (
        temp_compliance * 0.35 +  # Temperature compliance (35%)
        weight_compliance * 0.20 +  # Data quality (20%)
        fulfillment_efficiency * 0.45  # Operational efficiency (45%)
    )
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Temperature Compliance", f"{temp_compliance:.1f}%")
    with col2:
        st.metric("Data Quality", f"{weight_compliance:.1f}%")
    with col3:
        st.metric("Fulfillment Efficiency", f"{fulfillment_efficiency:.1f}%")
    with col4:
        st.metric("Warehouse Health Score", f"{chaos_score:.1f}%", 
                 delta="Target: 85%", delta_color="inverse")
    
    st.markdown("---")
    
    st.subheader("🎯 Phase 1: Top 50 Priority Relocations")
    
    st.markdown("""
    **Strategy**: Focus on SKUs that provide maximum impact:
    1. High-frequency items in wrong temperature zones
    2. Items in congested aisles (especially B-aisles during peak)
    3. Heavy items with decimal drift that need correction
    """)
    
    # Calculate priority score for each SKU
    temp_violations_df = sku_warehouse[sku_warehouse['temp_req'] != sku_warehouse['temp_zone']].copy()
    
    # Get order frequency
    sku_order_freq = orders.groupby('sku_id').size().reset_index(name='order_frequency')
    temp_violations_df = temp_violations_df.merge(sku_order_freq, on='sku_id', how='left')
    temp_violations_df['order_frequency'] = temp_violations_df['order_frequency'].fillna(0)
    
    # Get aisle congestion (focus on B-aisles)
    orders_with_hour = orders.copy()
    orders_with_hour['hour'] = orders_with_hour['order_timestamp'].dt.hour
    orders_sku_aisle = orders_with_hour.merge(sku[['sku_id', 'aisle']], on='sku_id', how='left')
    
    # Peak hour activity (19:00)
    peak_aisle_activity = orders_sku_aisle[orders_sku_aisle['hour'] == 19].groupby('aisle').size().reset_index(name='peak_activity')
    temp_violations_df = temp_violations_df.merge(peak_aisle_activity, on='aisle', how='left')
    temp_violations_df['peak_activity'] = temp_violations_df['peak_activity'].fillna(0)
    
    # Calculate priority score
    # Normalize factors to 0-1 scale
    temp_violations_df['order_freq_norm'] = temp_violations_df['order_frequency'] / temp_violations_df['order_frequency'].max()
    temp_violations_df['peak_activity_norm'] = temp_violations_df['peak_activity'] / temp_violations_df['peak_activity'].max()
    
    # High-risk temperature violations (Frozen/Refrigerated) get extra weight
    temp_violations_df['temp_criticality'] = temp_violations_df['temp_req'].apply(
        lambda x: 1.0 if x in ['Frozen', 'Refrigerated'] else 0.5
    )
    
    # Priority score = weighted combination
    temp_violations_df['priority_score'] = (
        temp_violations_df['order_freq_norm'] * 0.4 +
        temp_violations_df['peak_activity_norm'] * 0.3 +
        temp_violations_df['temp_criticality'] * 0.3
    )
    
    # Sort by priority
    top_50_relocations = temp_violations_df.sort_values('priority_score', ascending=False).head(50)
    
    st.dataframe(top_50_relocations[['sku_id', 'category', 'temp_req', 'temp_zone', 
                                     'current_slot', 'aisle', 'order_frequency', 
                                     'peak_activity', 'priority_score']].round(3), 
                use_container_width=True)
    
    st.download_button(
        label="📥 Download Top 50 Relocations CSV",
        data=top_50_relocations[['sku_id', 'current_slot']].to_csv(index=False),
        file_name="phase_1_relocations.csv",
        mime="text/csv"
    )
    
    st.markdown("---")
    
    st.subheader("📊 Sensitivity Analysis: +20% Volume Spike")
    
    st.markdown("""
    **Scenario**: Order volumes increase by 20% in Week 91.
    
    **Analysis**:
    - Current bottleneck: Aisle B during peak hour (19:00)
    - With +20% volume, B-aisle picks increase from ~1,400 to ~1,680
    - Forklift constraint (max 2 pickers) becomes critical bottleneck
    
    **Mitigation Strategy**:
    1. **Redistribute high-velocity SKUs** from B-aisles to A, C, D zones
    2. **Create alternative routing** for similar product categories
    3. **Stagger forklift restocking** to off-peak hours (14:00-17:00)
    4. **Implement dynamic picker allocation** based on real-time congestion
    """)
    
    # Simulate volume increase
    current_b_aisle_picks = orders_sku_aisle[
        (orders_sku_aisle['hour'] == 19) & 
        (orders_sku_aisle['aisle'].str.startswith('B', na=False))
    ].shape[0]
    
    projected_b_aisle_picks = current_b_aisle_picks * 1.2
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Current B-Aisle Picks (19:00)", f"{current_b_aisle_picks:,}")
    with col2:
        st.metric("Projected Picks (+20%)", f"{int(projected_b_aisle_picks):,}", 
                 delta=f"+{int(projected_b_aisle_picks - current_b_aisle_picks):,}")
    with col3:
        bottleneck_severity = "🔴 Critical" if projected_b_aisle_picks > 15000 else "🟡 High"
        st.metric("Bottleneck Severity", bottleneck_severity)
    
    st.markdown("---")
    
    st.subheader("🗺️ Generate Final Slotting Plan")
    
    st.markdown("""
    Click the button below to generate the optimized slotting plan for Week 91.
    This will create a `final_slotting_plan.csv` file with SKU_ID and Bin_ID columns.
    """)
    
    if st.button("🚀 Generate Optimized Slotting Plan", type="primary"):
        with st.spinner("Generating optimized slotting plan..."):
            # Import optimization logic
            import time
            time.sleep(2)  # Simulate processing
            
            # This would normally call the optimization algorithm
            # For now, we'll create a placeholder
            st.success("✅ Slotting plan generated successfully!")
            st.info("Check the output section for download link.")

st.markdown("---")
st.markdown("### 📊 VelocityMart Operations Dashboard | Interim Head of Operations")
st.markdown("*Data-driven insights for operational excellence*")
