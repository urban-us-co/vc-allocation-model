import streamlit as st
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import openai

# Set page configuration
st.set_page_config(page_title="Venture Fund Allocation Model", layout="wide")

# Initialize session state variables
if 'simulation_run' not in st.session_state:
    st.session_state.simulation_run = False
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ''

# OpenAI API Key Input (always visible in the sidebar)
st.sidebar.header("OpenAI API Key")
openai_api_key = st.sidebar.text_input(
    "Enter your OpenAI API key:",
    type="password",
    help="Your OpenAI API key is required to generate suggestions.",
    value=st.session_state.openai_api_key
)
st.session_state.openai_api_key = openai_api_key

# Customize inputs with default values and descriptions
st.sidebar.header("Venture Fund Parameters")

# Allow user to input capital raised
total_capital = st.sidebar.number_input(
    'Total Capital Raised (in $)',
    min_value=5_000_000,
    max_value=250_000_000,
    value=50_000_000,
    step=2_500_000,
    help="The total amount of capital raised for the venture fund."
)

# Calculate investment limits
max_investment_per_company = int(0.10 * total_capital)  # 10% of total capital
max_investment_per_period = int(0.20 * total_capital)   # 20% of total capital

# Display investment limits
st.sidebar.markdown(f"**Per-Company Investment Limit:** ${max_investment_per_company:,.0f}")
st.sidebar.markdown(f"**Per-Period Investment Limit:** ${max_investment_per_period:,.0f}")

# Fund Lifecycle Parameters
st.sidebar.header("Fund Lifecycle Parameters")

# Allow user to select initial investment period length
initial_investment_period = st.sidebar.slider(
    'Initial Investment Period (years)',
    min_value=1,
    max_value=5,
    value=3,
    help="Duration in years during which new investments are made."
)

# Total fund life is fixed at 12 years
total_fund_life = 12  # Total number of periods representing years

# Portfolio Parameters
st.sidebar.header("Portfolio Parameters")

# Allow user to set target number of new investments
target_new_investments = st.sidebar.number_input(
    'Target Number of New Investments',
    min_value=1,
    max_value=100,
    value=35,
    step=1,
    help="Total number of new investments to be made during the initial investment period."
)

# Capital Allocation
st.sidebar.header("Capital Allocation")

# Allow user to allocate capital between new investments and follow-ons
allocation_new_investments = st.sidebar.slider(
    'Capital Allocation to New Investments (%)',
    min_value=0,
    max_value=100,
    value=50,
    step=5,
    help="Percentage of total capital allocated to new investments."
)

allocation_follow_on_investments = 100 - allocation_new_investments
st.sidebar.write(f"Capital Allocation to Follow-On Investments (%): {allocation_follow_on_investments}")

# Initial Investment Strategy
st.sidebar.header("Initial Investment Strategy")

# Allow user to adjust the initial investment amount per company
initial_investment_amount = st.sidebar.number_input(
    'Initial Investment Amount per Company ($)',
    min_value=100_000,
    max_value=min(max_investment_per_company, total_capital),
    value=1_000_000,
    step=50_000,
    help=f"The amount invested in each company during the initial investment (Max: ${max_investment_per_company:,.0f})."
)

# Check if initial investment amount exceeds per-company limit
if initial_investment_amount > max_investment_per_company:
    st.sidebar.error(f"Initial investment amount cannot exceed ${max_investment_per_company:,.0f} (10% of total capital).")

# Follow-On Investment Strategy
st.sidebar.header("Follow-On Investment Strategy")

follow_on_strategy = st.sidebar.selectbox(
    'Follow-On Investment Strategy',
    options=[
        'Invest in High Success Probability Companies',
        'Invest in All Existing Portfolio Companies',
        'Selective Investment Based on Stage'
    ],
    help="Strategy for allocating follow-on investments."
)

# Allow user to adjust the follow-on investment amount per company
follow_on_investment_amount = st.sidebar.number_input(
    'Follow-On Investment Amount per Company ($)',
    min_value=100_000,
    max_value=min(max_investment_per_company, total_capital),
    value=500_000,
    step=50_000,
    help=f"The amount invested in each company during follow-on rounds (Max per company total investment: ${max_investment_per_company:,.0f})."
)

# Deal Terms Adjustments
st.sidebar.header("Deal Term Adjustments")

# Liquidation preferences multiplier
liquidation_preferences = st.sidebar.number_input(
    'Liquidation Preferences Multiplier',
    min_value=1.0,
    max_value=2.0,
    value=1.0,
    step=0.1,
    help="Multiplier for liquidation preferences (e.g., 1.5x preference)."
)

# Redemption rights influence
redemption_rights_influence = st.sidebar.slider(
    'Redemption Rights Influence on Returns',
    min_value=0.0,
    max_value=0.2,
    value=0.05,
    step=0.01,
    help="Increase in return due to redemption rights."
)

# Median post-money valuation
median_post_money_valuation = st.sidebar.number_input(
    'Median Post-Money Valuation ($)',
    min_value=5_000_000,
    max_value=100_000_000,
    value=10_000_000,
    step=1_000_000,
    help="Median post-money valuation for initial investments."
)

# Valuation cap discount
valuation_cap_discount = st.sidebar.slider(
    'Valuation Cap Discount',
    min_value=0.5,
    max_value=1.0,
    value=0.8,
    step=0.05,
    help="Discount for convertible securities (e.g., 0.8 for 20% discount)."
)

# "Trick Dice" Strategies Adjustments
st.sidebar.header("Strategic Adjustments")

# Sidebar inputs for better_deal_terms_multiplier
better_deal_terms_multiplier = st.sidebar.slider(
    'Better Deal Terms Multiplier',
    min_value=1.0,
    max_value=2.0,
    value=1.2,
    step=0.1,
    help="Multiplier to increase expected returns due to better deal terms."
)

# Sidebar inputs for active_involvement_success_increase
active_involvement_success_increase = st.sidebar.slider(
    'Active Involvement Success Increase',
    min_value=0.0,
    max_value=0.5,
    value=0.1,
    step=0.05,
    help="Increase in success probability due to active involvement."
)

# Sidebar inputs for network_exit_multiplier
network_exit_multiplier = st.sidebar.slider(
    'Network Exit Multiplier',
    min_value=1.0,
    max_value=2.0,
    value=1.1,
    step=0.1,
    help="Multiplier to increase exit valuations due to network effects."
)

# VC Transaction Data
vc_transaction_data = {
    'liquidation_preferences': liquidation_preferences,
    'redemption_rights_influence': redemption_rights_influence,
    'median_post_money_valuation': median_post_money_valuation,
    'valuation_cap_discount': valuation_cap_discount,
}

# Outcome Probabilities and Returns
st.sidebar.header("Outcome Parameters")

# Adjusting the outcome multiples to target 3-4x MOIC
low_return_multiple = st.sidebar.number_input(
    'Low Return Multiple',
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.1,
    help="Multiple for low return outcomes."
)

medium_return_multiple = st.sidebar.number_input(
    'Medium Return Multiple',
    min_value=1.0,
    max_value=3.0,
    value=1.0,
    step=0.5,
    help="Multiple for medium return outcomes."
)

high_return_multiple = st.sidebar.number_input(
    'High Return Multiple',
    min_value=3.0,
    max_value=10.0,
    value=5.0,
    step=0.5,
    help="Multiple for high return outcomes."
)

very_high_return_multiple = st.sidebar.number_input(
    'Very High Return Multiple',
    min_value=10.0,
    max_value=30.0,
    value=10.0,
    step=1.0,
    help="Multiple for very high return outcomes."
)

extremely_high_return_multiple = st.sidebar.number_input(
    'Extremely High Return Multiple',
    min_value=20.0,
    max_value=100.0,
    value=20.0,
    step=5.0,
    help="Multiple for extremely high return outcomes."
)

# Outcome Probabilities and Returns
outcomes_dict = {
    'Fund Impact': ['LOW', 'MEDIUM', 'HIGH', 'VERY HIGH', 'EXTREMELY HIGH'],
    'Outcome Metric': [
        low_return_multiple,
        medium_return_multiple,
        high_return_multiple,
        very_high_return_multiple,
        extremely_high_return_multiple
    ],
    'Probability': [0.5, 0.3, 0.1, 0.05, 0.05]  # Probabilities
}
outcomes_df = pd.DataFrame(outcomes_dict)

# Adjust Outcomes Function
def adjust_outcomes(df):
    adjusted_df = df.copy()
    # Adjust returns for better deal terms and network exit multiplier
    adjusted_df['Adjusted Outcome Metric'] = (
        adjusted_df['Outcome Metric'] *
        better_deal_terms_multiplier *
        network_exit_multiplier *
        vc_transaction_data['valuation_cap_discount']
    )
    # Adjust for liquidation preferences and redemption rights
    adjusted_df['Adjusted Outcome Metric'] *= vc_transaction_data['liquidation_preferences']
    adjusted_df['Adjusted Outcome Metric'] += vc_transaction_data['redemption_rights_influence'] * adjusted_df['Outcome Metric']
    # Adjusted probabilities remain the same
    adjusted_df['Adjusted Probability'] = adjusted_df['Probability']
    # Normalize probabilities
    total_prob = adjusted_df['Adjusted Probability'].sum()
    adjusted_df['Adjusted Probability'] /= total_prob
    return adjusted_df

# Apply the function to adjust outcomes
adjusted_outcomes_df = adjust_outcomes(outcomes_df)

# Define market condition variables
market_conditions = ['Bull Market', 'Neutral Market', 'Bear Market']
market_probabilities = [0.3, 0.5, 0.2]  # Probabilities summing to 1

# Define company growth rate variables
growth_rates = ['High Growth', 'Moderate Growth', 'Low Growth']
growth_probabilities = [0.4, 0.4, 0.2]  # Probabilities summing to 1

# Classes
class Investment:
    def __init__(self, amount, stage=1, success_prob=0.1):
        self.amount = amount
        self.stage = stage
        self.success_prob = success_prob
        self.is_active = True

    def __repr__(self):
        return f"Investment(amount={self.amount}, stage={self.stage}, success_prob={self.success_prob:.2f})"

class State:
    def __init__(self, capital, time, portfolio):
        self.capital = capital
        self.time = time
        self.portfolio = portfolio

    def __repr__(self):
        return f"State(time={self.time}, capital={self.capital}, portfolio_size={len(self.portfolio)})"

class Action:
    def __init__(self, new_investments=None, follow_on_investments=None, exits=None):
        self.new_investments = new_investments if new_investments is not None else []
        self.follow_on_investments = follow_on_investments if follow_on_investments is not None else []
        self.exits = exits if exits is not None else []

    def __repr__(self):
        return f"Action(new={len(self.new_investments)}, follow_on={len(self.follow_on_investments)}, exits={len(self.exits)})"

# Reward and Transition Functions
def reward(state, action):
    immediate_reward = 0
    # Handle exits
    for investment in action.exits:
        if investment.is_active:
            # Determine outcome
            outcome = np.random.choice(
                adjusted_outcomes_df['Adjusted Outcome Metric'],
                p=adjusted_outcomes_df['Adjusted Probability']
            )
            immediate_reward += investment.amount * outcome
            investment.is_active = False
            # Remove from portfolio
            if investment in state.portfolio:
                state.portfolio.remove(investment)
    # Subtract new investment amounts
    total_new_investment = sum(inv.amount for inv in action.new_investments)
    total_follow_on_investment = sum(inv.amount for inv in action.follow_on_investments)
    total_invested = total_new_investment + total_follow_on_investment

    # Enforce per-period investment limit
    if total_invested > max_investment_per_period:
        excess = total_invested - max_investment_per_period
        immediate_reward += excess  # Refund the excess investment
        total_invested = max_investment_per_period
        st.warning(f"Per-period investment limit exceeded. Excess of ${excess:,.0f} not invested.")

    immediate_reward -= total_invested
    return immediate_reward

def transition(state, action):
    # Update capital
    total_invested = sum(inv.amount for inv in action.new_investments + action.follow_on_investments)
    # Enforce per-period investment limit
    if total_invested > max_investment_per_period:
        total_invested = max_investment_per_period
    next_capital = state.capital - total_invested
    # Update time
    next_time = state.time + 1
    # Update portfolio
    next_portfolio = state.portfolio.copy()
    # Add new investments
    next_portfolio.extend(action.new_investments)
    # Update follow-on investments
    for inv in action.follow_on_investments:
        inv.stage += 1
        inv.success_prob = min(inv.success_prob + active_involvement_success_increase, 1.0)
        if inv not in next_portfolio:
            next_portfolio.append(inv)
    # Update existing investments
    for inv in next_portfolio:
        if inv.is_active and inv not in action.follow_on_investments:
            inv.stage += 1
    # Return new state
    return State(next_capital, next_time, next_portfolio)

# Investment Policy Function
def investment_policy(state):
    action = Action()
    total_invested_this_period = 0

    # New Investments during initial investment period
    if state.time < initial_investment_period and state.capital > 0:
        # Determine number of new investments to make this period
        remaining_investments = target_new_investments - len([inv for inv in state.portfolio if inv.stage == 1])
        avg_investments_per_period = target_new_investments / initial_investment_period
        investments_this_period = min(np.random.poisson(avg_investments_per_period), remaining_investments)
        # Ensure we have enough capital
        max_investments_affordable = int(state.capital // initial_investment_amount)
        investments_this_period = min(investments_this_period, max_investments_affordable)
        # Ensure per-period investment limit is not exceeded
        total_potential_investment = investments_this_period * initial_investment_amount
        if total_invested_this_period + total_potential_investment > max_investment_per_period:
            available_investment = max_investment_per_period - total_invested_this_period
            investments_this_period = int(available_investment // initial_investment_amount)
        if investments_this_period > 0:
            new_investments = []
            for _ in range(investments_this_period):
                # Ensure per-company investment limit is not exceeded
                total_invested_in_company = initial_investment_amount
                if total_invested_in_company <= max_investment_per_company:
                    new_investments.append(Investment(amount=initial_investment_amount))
            action.new_investments.extend(new_investments)
            total_invested_this_period += len(new_investments) * initial_investment_amount

    # Follow-On Investments
    if state.capital > 0 and state.portfolio:
        # Decide which companies to invest in based on strategy
        if follow_on_strategy == 'Invest in High Success Probability Companies':
            candidates = [inv for inv in state.portfolio if inv.is_active and inv.success_prob >= 0.5]
        elif follow_on_strategy == 'Invest in All Existing Portfolio Companies':
            candidates = [inv for inv in state.portfolio if inv.is_active]
        elif follow_on_strategy == 'Selective Investment Based on Stage':
            candidates = [inv for inv in state.portfolio if inv.is_active and inv.stage >= 2]
        else:
            candidates = []
        # Ensure per-period investment limit is not exceeded
        available_investment = max_investment_per_period - total_invested_this_period
        max_follow_on_affordable = int(min(state.capital, available_investment) // follow_on_investment_amount)
        num_follow_on = min(len(candidates), max_follow_on_affordable)
        if num_follow_on > 0:
            follow_on_investments = []
            for inv in random.sample(candidates, num_follow_on):
                # Ensure per-company investment limit is not exceeded
                total_invested_in_company = inv.amount + follow_on_investment_amount
                if total_invested_in_company <= max_investment_per_company:
                    inv.amount += follow_on_investment_amount  # Increase investment amount
                    follow_on_investments.append(inv)
                    total_invested_this_period += follow_on_investment_amount
            action.follow_on_investments.extend(follow_on_investments)
    # Exits
    exits = []
    for inv in state.portfolio:
        # Determine if the investment exits this period
        exit_probability = 0.05  # 5% chance of exit per period
        if inv.is_active and random.random() < exit_probability:
            exits.append(inv)
    action.exits.extend(exits)
    return action

# Simulate policy and returns
def simulate_policy(simulations=1000):
    moic_results = []
    for _ in range(simulations):
        # Initialize state
        state = State(
            capital=total_capital,
            time=0,
            portfolio=[]
        )
        cumulative_return = 0
        total_invested = 0
        total_returned = 0
        while state.time < total_fund_life:
            action = investment_policy(state)
            immediate_reward = reward(state, action)
            cumulative_return += immediate_reward
            # Update total invested and returned
            invested_capital = sum(inv.amount for inv in action.new_investments + action.follow_on_investments)
            returned_capital = sum(inv.amount * adjusted_outcomes_df['Adjusted Outcome Metric'].mean() for inv in action.exits)
            total_invested += invested_capital
            total_returned += returned_capital
            state = transition(state, action)
        # Handle remaining portfolio exits at the end of fund life
        for investment in state.portfolio:
            if investment.is_active:
                # Determine outcome
                outcome = np.random.choice(
                    adjusted_outcomes_df['Adjusted Outcome Metric'],
                    p=adjusted_outcomes_df['Adjusted Probability']
                )
                exit_return = investment.amount * outcome
                cumulative_return += exit_return
                total_returned += exit_return
                total_invested += investment.amount  # Include any remaining invested capital
                investment.is_active = False
        # Calculate MOIC
        moic = total_returned / total_capital if total_capital > 0 else 0
        moic_results.append(moic)
    return moic_results


# Streamlit UI
st.title('Venture Fund Allocation Model')
st.write("""
    This app simulates the impact of various investment strategies and deal terms on the performance 
    of venture investments over a typical fund lifecycle. The focus is on calculating 
    the Multiple on Invested Capital (MOIC) for the entire fund.
""")

# Run Simulation Button
if st.button('Run Simulation'):
    with st.spinner('Running simulations...'):
        moic_results = simulate_policy()
    mean_moic = np.mean(moic_results)
    median_moic = np.median(moic_results)
    std_moic = np.std(moic_results)
    
    # Store results in session state
    st.session_state.moic_results = moic_results
    st.session_state.mean_moic = mean_moic
    st.session_state.median_moic = median_moic
    st.session_state.std_moic = std_moic
    st.session_state.simulation_run = True

# Display simulation results if they exist
if st.session_state.simulation_run:
    st.subheader("Simulation Results:")
    st.write(f"**Mean MOIC:** {st.session_state.mean_moic:.2f}x")
    st.write(f"**Median MOIC:** {st.session_state.median_moic:.2f}x")
    st.write(f"**Standard Deviation of MOIC:** {st.session_state.std_moic:.2f}x")
    
    # Plot histogram
    fig, ax = plt.subplots()
    ax.hist(st.session_state.moic_results, bins=30, edgecolor='k')
    ax.set_title('Distribution of Simulated MOIC for the Entire Fund')
    ax.set_xlabel('MOIC (Multiple on Invested Capital)')
    ax.set_ylabel('Frequency')
    st.pyplot(fig)

    # Button to get suggestions
    if st.button('Get Suggestions to Improve MOIC'):
        if st.session_state.openai_api_key:
            openai.api_key = st.session_state.openai_api_key
            with st.spinner('Generating suggestions...'):
                # Collect all user inputs and simulation outcomes
                def generate_prompt():
                    prompt = f"""
I am running a venture fund simulation with the following parameters:

- **Venture Fund Parameters:**
  - Total Capital Raised: ${total_capital}
  - Initial Investment Period: {initial_investment_period} years
  - Total Fund Life: {total_fund_life} years

- **Portfolio Parameters:**
  - Target Number of New Investments: {target_new_investments}

- **Capital Allocation:**
  - Allocation to New Investments: {allocation_new_investments}%
  - Allocation to Follow-On Investments: {allocation_follow_on_investments}%

- **Investment Strategies:**
  - Initial Investment Amount per Company: ${initial_investment_amount}
  - Follow-On Investment Strategy: {follow_on_strategy}
  - Follow-On Investment Amount per Company: ${follow_on_investment_amount}

- **Deal Term Adjustments:**
  - Liquidation Preferences Multiplier: {liquidation_preferences}x
  - Redemption Rights Influence: {redemption_rights_influence}
  - Median Post-Money Valuation: ${median_post_money_valuation}
  - Valuation Cap Discount: {valuation_cap_discount}

- **Strategic Adjustments:**
  - Better Deal Terms Multiplier: {better_deal_terms_multiplier}
  - Active Involvement Success Increase: {active_involvement_success_increase}
  - Network Exit Multiplier: {network_exit_multiplier}

- **Outcome Parameters:**
  - Low Return Multiple: {low_return_multiple}x
  - Medium Return Multiple: {medium_return_multiple}x
  - High Return Multiple: {high_return_multiple}x
  - Very High Return Multiple: {very_high_return_multiple}x
  - Extremely High Return Multiple: {extremely_high_return_multiple}x
  - Outcome Probabilities: {outcomes_df['Probability'].tolist()}

- **Investment Limits:**
  - Per-Company Investment Limit: ${max_investment_per_company}
  - Per-Period Investment Limit: ${max_investment_per_period}

**Simulation Outcomes:**

- **Mean MOIC:** {st.session_state.mean_moic:.2f}x
- **Median MOIC:** {st.session_state.median_moic:.2f}x
- **Standard Deviation of MOIC:** {st.session_state.std_moic:.2f}x

**Typical Fund Goal:**

- The typical venture fund aims to achieve a 3-4x gross MOIC.

**Objective:**

Given that the current mean MOIC of my simulation is {st.session_state.mean_moic:.2f}x, which is below the typical fund goal, I would like suggestions on how to improve my model to reach or exceed this target.

**Request:**

Please provide recommendations on adjusting my investment strategies, deal terms, or other parameters that could enhance the fund's performance. Specifically, I'm interested in:

- Identifying parameters with the most significant impact on increasing the MOIC.
- Strategies to optimize capital allocation and investment amounts.
- Adjustments to outcome probabilities and multiples for a more favorable return distribution.
- Any other insights or best practices to achieve the desired 3-4x gross MOIC.

Thank you for your assistance.
"""
                    return prompt

                # Generate the prompt
                prompt = generate_prompt()

                # Use OpenAI API to get suggestions
                try:
                    response = openai.ChatCompletion.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=500,
                        n=1,
                        stop=None,
                        temperature=0.7,
                    )

                    suggestions = response.choices[0].message['content']

                    # Display the suggestions
                    st.subheader("Suggestions to Improve MOIC:")
                    st.write(suggestions)
                except Exception as e:
                    st.error(f"An error occurred while generating suggestions: {e}")
        else:
            st.warning("Please enter your OpenAI API key in the sidebar to generate suggestions.")

else:
    st.info("Please run the simulation to get results and suggestions.")