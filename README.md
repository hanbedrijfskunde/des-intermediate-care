# Discrete Event Simulation for Intermediate Care (Arnhem & Nijmegen Region)

## Introduction and Objectives

In October 2023, about 22,218 older adults were waiting to be placed in a nursing home in the Netherlands. In addition, 11,835 people were on precautionary waiting lists (used for emergency placements) despite the availability of crisis beds. It has also become increasingly difficult for seniors to obtain a place in a nursing home of their choice, and the eldercare sector faces significant staffing shortages. The combination of limited preferred placement options and workforce shortages has led to substantial queues for elderly care services **(ActiZ, n.d.)**. These long waits often mean that patients remain in hospitals longer than necessary while awaiting a nursing home bed, leading to adverse events and higher healthcare costs.

To help alleviate this issue, the Dutch government introduced **Intermediate Care** (known as Eerstelijns Verblijf, ELV, in Dutch). Intermediate Care provides short-term, bed-based care for health problems that do not require hospital admission but cannot be managed safely at home. **Melis et al. (2004)** describe Intermediate Care as a range of services designed to facilitate the transition from hospital to home and to promote patients’ functional independence. In the Netherlands, the goal of Intermediate Care is to enable older adults to recover and eventually return to living independently in the community **(Thijssen, 2023)**. However, the exact definition and scope of Intermediate Care are not uniform across studies and settings, making it difficult to standardize the concept **(Melis et al., 2004)**. **Steiner (2001)** evaluated several models of Intermediate Care – including hospital admission avoidance and post-acute rehabilitation – and concluded that Intermediate Care can indeed help reduce unnecessary hospital stays and expedite discharges, but that its full benefit will only be realized with appropriate implementation and sufficient resources.

One recent initiative to improve eldercare capacity and reduce waiting times is the **Dolce Vita project** (“Data-Driven Optimization for a Vital Elderly Care System in the Netherlands”). Launched in 2019, this project is a collaboration between the Centrum Wiskunde & Informatica, Vrije Universiteit Amsterdam, and Amsterdam UMC, aiming to optimize elderly care processes across multiple regions by modeling the entire chain of care from hospital to home **(Thijssen, 2023)**. The Twente region is one of the areas participating in the Dolce Vita project **(Thijssen, 2023)**. By mapping out the patient flow – from an acute hospital visit through Intermediate Care and ultimately back home – the project seeks to identify bottlenecks and test potential improvements. According to **CWI (n.d.)**, early results from this data-driven approach suggest that a new mathematical allocation model could potentially **halve the waiting lists** for nursing home placements in the Netherlands.

Apart from such nationwide initiatives, research has explored specific strategies to make patient flow more efficient. One proposed approach is **preference-based allocation** of patients to care facilities. **Arntzen et al. (2022)** introduced a model that assigns nursing home placements based on each patient’s preferences (for example, preferred location) rather than on a simple first-come-first-served basis. They found that accounting for individual preferences can substantially shorten waiting times for nursing home admission. A subsequent study by **Arntzen et al. (2024)** refined this allocation model and further demonstrated the benefits of incorporating patient preferences into the placement process. In parallel, **Arntzen et al. (2023)** used a simulation model of Intermediate Care in the Amsterdam region to show that improving access to Intermediate Care beds can help **avoid hospital admissions and reduce delayed discharges** (i.e. shorten the time patients wait in hospital for a post-acute care bed). This evidence from Amsterdam indicates that enhancing Intermediate Care capacity and efficiency can directly relieve pressure on hospitals.

Another important factor in reducing queues is the flexible use of available care capacity. In the hospital context, studies show that pooling or sharing beds among departments can improve efficiency. For example, **Bekker et al. (2017)** found that dynamically allocating beds across hospital wards (instead of strictly assigning fixed beds to each ward) reduced patient waiting times and improved overall flow. A similar principle could be applied to Intermediate Care by allowing beds to be shared between different care categories (such as rehabilitation and general care) as demand fluctuates. In addition, understanding the demand for post-hospital care is crucial. **de Groot et al. (2023)** reported that 12.2% of older patients discharged from an acute hospital in their study were referred to a rehabilitation-oriented care facility. This finding highlights a substantial need for Intermediate Care or rehabilitation beds following hospital stays. If sufficient Intermediate Care capacity is not available for this portion of patients, hospital discharges may be delayed—contributing to longer waits for incoming patients and compounding the overall waiting list problem.

A recent simulation case study in the Twente region evaluated several interventions to address these challenges **(van Loon, 2024)**. In that study, **van Loon (2024)** examined measures such as partial bed-sharing between care types, centralizing admissions across locations, prioritizing certain hospital referrals, and extending admission hours to include weekends. The results showed that a **partial bed-sharing** arrangement – specifically, making 40% of geriatric rehabilitation beds available to high-complex care patients (and vice versa) – could reduce the average waiting time by roughly **7.8%** (approximately 3.8 hours) **(van Loon, 2024)**. Additionally, allowing admissions on weekends (rather than only on weekdays) further decreased patient waiting times **(van Loon, 2024)**. These outcomes are consistent with trends observed in the Amsterdam simulation, which also found improvements in patient flow with increased flexibility and access **(Arntzen et al., 2023)**. Taken together, these findings underscore that targeted operational changes—such as smarter patient allocation and more flexible use of beds—can significantly improve efficiency in elderly care and help reduce waiting times for nursing home and Intermediate Care placements.

## Simulation Model Design

Our DES model represents the day-to-day operations of intermediate care facilities in the region. It simulates the journey of each patient from arrival (referral or hospital discharge) through triage and admission into an available bed, until discharge from the intermediate care. Key components and assumptions of the model include:

- **Patient Arrivals (Demand):** We assume **Poisson-distributed patient arrivals**, a common and validated approach for modeling random healthcare demand. Separate Poisson streams can represent different sources (e.g. hospital referrals, GP referrals for ELV, and emergency/crisis cases). The model can incorporate time-dependent arrival rates to reflect demand surges – for example, higher arrivals during daytime or weekday peaks. These surges capture the fluctuating demand that leads to periods of strain on bed availability.

- **Bed Capacity and Types:** The region’s intermediate care beds are categorized into regular nursing/ELV beds and a small number of **crisis beds** reserved for urgent cases. We initialize the model with the current number of beds in each category (based on Arnhem/Nijmegen data or estimates). Bed availability fluctuates as patients occupy and vacate beds over time. At baseline, capacity is tight – intermediate care units often operate at ~85% occupancy on average – meaning even modest demand spikes can fill all beds. The simulation tracks bed occupancy and any queue of waiting patients when demand temporarily exceeds supply.

- **Triage and Transfer Process:** When a patient is referred to intermediate care, an **admission/triage process** takes place before the patient actually transfers into a bed. In current practice, this involves administrative steps and finding a suitable placement, which introduces a delay (transfer time). We model this **triage delay** as an additional waiting time for the patient prior to bed admission. Based on expert input from similar cases, we use an average transfer delay of around 1.5 days in the baseline scenario. This reflects the observed inefficiencies in processing referrals and coordinating handovers. During this period, a patient may effectively hold a spot in queue (or even stay in hospital if coming from an acute setting). If a bed is free but triage isn’t completed, the bed might appear “available” but cannot be filled until the process finishes. This component is crucial, as **inefficient triage contributes significantly to overall waiting times**.

- **Daily Operations and Events:** The DES model moves through time, logging events such as patient arrivals, admissions, and discharges. We simulate at the granularity of hours, but summarize outputs per day for reporting. Each day:
  - New patient arrivals occur (randomly via the Poisson process).
  - If a patient arrives and a suitable bed is _immediately available_ **and** the facility is accepting admissions at that time, the patient begins the transfer/triage process or is admitted straight away if triage is instantaneous. Otherwise, the patient joins a waiting list.
  - Triage completion events: after the simulated triage delay, waiting patients can be assigned to the next available bed.
  - Discharges: Patients currently in intermediate care complete their stay based on a **length-of-stay (LOS)** distribution (modeled as exponential or gamma, reflecting observed stay lengths. When a patient is discharged, that bed becomes free for the next waiting patient (if any).
  - The model accounts for day/night and weekday/weekend differences. For example, in the baseline, admissions may be restricted at night or on weekends due to limited staffing (any arrivals during off-hours would have to wait until the next morning or Monday). These operational rules are captured in the simulation schedule.

```mermaid
flowchart TD
    StartPatient([Patient Arrives]) --> CheckAdm{Admissions Allowed?}
        CheckAdm -- Yes --> RequestBed[Request Appropriate Bed]
        CheckAdm -- No --> WaitAdm[Wait Until Next Admission Window]
        WaitAdm --> RequestBed

        RequestBed --> BedAvail{Bed Available?}
        BedAvail -- No --> WaitQueue[Wait in Queue]
        WaitQueue --> BedAvail

        BedAvail -- Yes --> StartTriage[Start Triage Process]
        StartTriage --> WaitTriage[Wait for Triage Delay]
        WaitTriage --> Admit[Admit Patient to Bed]

        Admit --> RecordWait[Record Waiting Time]
        RecordWait --> GenLOS[Generate Length of Stay]
        GenLOS --> OccupyBed[Occupy Bed for LOS Duration]
        OccupyBed --> Discharge[Discharge Patient]
        Discharge --> RecordDisch[Record Discharge]
```

By simulating many days (e.g. over several months or a year with multiple replications), we obtain statistically reliable performance metrics. The model is calibrated to current conditions so that baseline results (occupancy, average waits, etc.) align with known data or estimates for the Arnhem and Nijmegen region.

## Policy Scenarios Tested

We use the DES model to experiment with several **policy scenarios** and compare them to the status quo. Each scenario modifies the model’s parameters or logic to reflect a proposed improvement in intermediate care operations:

- **1. 24/7 Admissions:** In this scenario, intermediate care facilities accept admissions around the clock, including evenings and weekends. We remove the constraint that prevented admissions outside of normal working hours. Concretely, patients can be transferred to a bed as soon as one is free and triage is done, even at 2 AM or on a Sunday. This policy prevents the build-up of a weekend backlog and avoids situations where patients wait idle simply because the intake office is closed. A similar approach was studied as an “admission turns” system where facilities take evening/night admission duty by rotation, effectively enabling 24/7 access with minimal delays. We expect this change to **significantly reduce waiting times**, especially for patients whose referrals currently arrive late in the day or on Fridays. By eliminating off-hour delays, the flow of patients becomes smoother. Prior research indicates that enabling evening/weekend admissions can sharply cut wait times and prevent unnecessary hospital days . Our simulation will show if freeing up admissions timing leads to higher bed utilization (by filling beds faster) without overwhelming the system.

- **2. Improved Triage Efficiency:** This scenario streamlines the admission process to **reduce transfer times**. We simulate a more efficient triage system – for example, dedicating staff to quickly process referrals or implementing a centralized placement system – such that the administrative delay is shortened (e.g. from ~1.5 days to a few hours). In the model, we might reduce the average triage delay to, say, 4 hours (as in the cited admission-turns scenario or even approach zero delay in an ideal case. The impact of this is that patients move into available beds faster, freeing up hospital resources and improving flow. We anticipate a substantial drop in waiting time as this was identified as a key bottleneck. Indeed, a sensitivity analysis in Amsterdam showed that the prevailing ~1.8-day wait was largely caused by the triage/transfer lag and could be almost eliminated by speeding up admissions. Our DES results should reflect that **faster triage leads to shorter queues** and fewer patients held up in hospital. It may also slightly increase overall throughput (more admissions processed in a given time) if beds were occasionally left idle waiting for paperwork in the baseline. We will observe metrics like the average transfer delay and how many patients are waiting at any time under this improved process.

- **3. Additional Bed Capacity:** This scenario tests the effect of increasing the number of intermediate care beds available. We simulate an expansion of capacity (for example, adding a certain number of ELV beds or opening a new wing). Intuitively, more beds should accommodate more patients and reduce competition for space. This is especially helpful during **demand surges** – sudden influxes of patients can be absorbed if spare beds are available, rather than causing a queue. The model will allow us to adjust total bed count and examine resulting occupancy and wait times. However, it is important to note that simply adding beds may have **diminishing returns** if the core delays are elsewhere. If, for instance, triage is very slow or admissions are limited by scheduling, extra beds could sit empty waiting for administrative processes to catch up. A prior case study found that increasing beds beyond the current level did not significantly reduce waiting time when triage and off-hour closures were the real bottleneck . Therefore, our simulation will highlight whether the Arnhem/Nijmegen region is truly capacity-constrained or if operational efficiencies yield more benefit. We will run scenarios with, say, +10% and +20% bed capacity and observe the changes in occupancy rate and queue lengths.

We can also combine these interventions (for example, implementing both 24/7 admissions and improved triage together) to see **synergistic effects**. In fact, combining process improvements has the potential to nearly eliminate waits – one study showed that opening intermediate care 24/7 _and_ expediting transfers brought the average waiting time to virtually zero. Our scenario analysis will compare the status quo against each policy (and combinations, if relevant) to identify which changes yield the largest improvements in access and flow.

## Simulation Outputs and Dashboard

After running the simulation for each scenario, we collect key **performance metrics** to evaluate intermediate care availability and responsiveness. These metrics are presented through a visual dashboard and accompanying statistical reports for clarity. The dashboard would likely include charts and tables for the following indicators:

- **Bed Availability & Occupancy:** We track the number of beds occupied vs. free each day. The dashboard can show the average **occupancy rate** (percentage of beds in use) as well as trends over time (peaks and troughs in bed usage). High occupancy (e.g. consistently >85% indicates strain on the system. The simulation output might highlight how often the system is at full capacity and for how long. For each scenario, we compare occupancy levels – for instance, additional bed capacity might lower the average occupancy, whereas 24/7 admissions might keep beds from sitting idle overnight, slightly increasing average occupancy but reducing waiting.

- **Waiting Times for Admission:** This is a critical metric representing how long patients wait from the time they are ready for intermediate care (e.g. hospital discharge or GP referral) until they are actually admitted to a bed. We report the average waiting time and distribution (e.g. median and percent of patients waiting more than 1 day, etc.) for each scenario. The dashboard could show a **comparison of waiting times** in a bar chart for baseline vs. each policy scenario. We expect to see markedly lower waiting times under the 24/7 admission and improved triage scenarios. For example, in the efficient triage + extended hours scenario, the model may show wait times dropping to only a few hours, whereas the baseline might be over a day. Shorter waits mean fewer delayed transfers of care, which translates to fewer unnecessary hospital bed days.

- **Throughput and Admissions:** The reports will include the total number of patients admitted per day or week. This reflects how well the system is handling demand. If a scenario enables more admissions (by reducing downtime or freeing capacity), we will see an increase in throughput. For instance, allowing weekend admissions adds two extra days of intake, so weekly admissions could increase accordingly. We also monitor if any patients **could not be admitted** (or had to be diverted) due to lack of beds – a crucial indicator of system capacity shortfall. Ideally, with sufficient beds or better processes, denied admissions should approach zero.

- **Crisis vs Routine Admissions:** Since our model includes crisis beds, we can generate stats specific to urgent cases. The dashboard might show the average wait for crisis cases (which should be minimal by design) and how often a crisis patient finds no immediate bed. This helps ensure that emergency needs are met. Similarly, we can report separate occupancy for crisis-designated beds versus regular ELV beds, if applicable.

- **Comparative Scenario Impact:** A special section of the dashboard or report will summarize **the impact of each policy scenario** side by side. For example, a table might list each scenario (24/7 admissions, Improved Triage, Added Beds, and perhaps combinations) and key outcomes: average wait time, occupancy rate, percentage of days with full occupancy, etc. This makes it easy to identify which intervention yields the best improvement. We expect to see that process-oriented interventions (triage and 24/7 access) have a dramatic effect on waiting time and slightly improved throughput, whereas adding beds might show moderate reductions in wait time unless demand was extremely high. Such comparative visuals and statistics provide evidence for decision-makers on which changes would most improve intermediate care performance.

```mermaid
flowchart TD
    Start([Start Simulation]) --> InitEnv[Initialize Environment & Resources]
    InitEnv --> InitStats[Initialize Statistics Collection]
    InitStats --> StartProc[Start Processes]

    StartProc --> RegPatGen[Regular Patient Generator]
    StartProc --> CrisisPatGen[Crisis Patient Generator]
    StartProc --> StatsCol[Statistics Collection Process]
    StartProc --> RunSim[Run Simulation for Duration]

    subgraph PatientGenerators["Patient Generators"]
        RegPatGen --> CalcArrival1[Calculate Next Arrival Time]
        CalcArrival1 --> WaitArrival1[Wait Until Next Arrival]
        WaitArrival1 --> CreatePat1[Create Regular Patient]
        CreatePat1 --> LogArr1[Log Arrival]
        LogArr1 --> StartProc1[Start Patient Process]
        StartProc1 --> CalcArrival1

        CrisisPatGen --> CalcArrival2[Calculate Next Arrival Time]
        CalcArrival2 --> WaitArrival2[Wait Until Next Arrival]
        WaitArrival2 --> CreatePat2[Create Crisis Patient]
        CreatePat2 --> LogArr2[Log Arrival]
        LogArr2 --> StartProc2[Start Patient Process]
        StartProc2 --> CalcArrival2
    end

    subgraph StatisticsCollection["Statistics Collection"]
        StatsCol --> WaitHour[Wait for Next Hour]
        WaitHour --> RecordOcc[Record Current Occupancy]
        RecordOcc --> RecordQueue[Record Queue Lengths]
        RecordQueue --> WaitHour
    end

    RunSim --> ProcessResults[Process Results]
    ProcessResults --> CreateVis[Create Visualizations]
    CreateVis --> End([End Simulation])

    classDef processNode fill:#f9f,stroke:#333,stroke-width:2px
    classDef decisionNode fill:#bbf,stroke:#333,stroke-width:2px
    classDef startEndNode fill:#9f9,stroke:#333,stroke-width:2px

    class Start,End startEndNode
    class CheckAdm,BedAvail decisionNode
    class InitEnv,InitStats,StartProc,RunSim,ProcessResults,CreateVis processNode
```

The dashboard would be interactive and visual, but since we cannot render charts here, one can imagine line graphs of occupancy over time, bar charts of average wait by scenario, and perhaps distribution plots of wait times. All these outputs together give a comprehensive view of the system’s behavior under each policy.

## Insights and Recommendations

By analyzing the simulation results, we gain valuable insights into how to **optimize intermediate care in the Arnhem and Nijmegen region**. The DES model highlights where the true bottlenecks lie: whether in capacity or in operational processes. In similar case studies, it was found that long waits were **not always due to insufficient beds, but rather due to inefficiencies in admissions and triage**. Our scenario tests will likely reflect the same pattern. Key findings and recommendations might include:

- **Triage Efficiency is Critical:** Streamlining the referral and triage process can yield significant reductions in waiting time and improve patient flow. The simulation shows that even without adding new beds, addressing the administrative delays (for example, by introducing a faster triage protocol or a dedicated intake team) can utilize existing capacity more effectively. This aligns with evidence that an inefficient application process was the main cause of delays in intermediate care access. We recommend prioritizing investments in a more efficient triage system (e.g. centralized intake software, better coordination between hospitals and nursing homes) as a first step to cut down wait times.

- **Extend Admission Hours:** Enabling **24/7 admissions** (or at least evening and weekend intake) keeps the intermediate care pipeline flowing continuously. The DES scenario demonstrates that when admissions are not restricted to office hours, patients no longer accumulate in queues over the weekend, and beds that free up on a Friday night can be filled by Saturday instead of remaining empty till Monday. This policy can drastically reduce the average wait and virtually eliminate the weekend effect. Regions that piloted round-the-clock admission coverage saw waiting times drop to near zero, indicating that this measure can effectively solve delays when coupled with efficient triage. Therefore, we recommend implementing an on-call admission rotation or staffing plan to allow admissions at any time, ensuring no avoidable delay for patients ready to transfer.

- **Adequate (but Not Excessive) Bed Capacity:** While adding beds alone is not a silver bullet, it can provide a buffer during peak demand. The simulation results will show how much waiting time improves with additional capacity. If the baseline model exhibits frequent periods where all beds are occupied and a queue forms, that signals a need for capacity expansion in the region. On the other hand, if waits drop mainly through process improvements, it underlines that existing capacity was underutilized due to logistic issues. Our model can identify the **optimal number of beds** needed to meet demand without excessive idle capacity. For example, if raising capacity by 20% yields no waiting at all, one might choose a smaller increase that balances cost vs. benefit (since extremely low occupancy is inefficient). Decision-makers can use these insights to plan investments in new beds or redistribution of beds across facilities in Arnhem and Nijmegen.

- **Impact on Hospitals:** A crucial outcome of optimizing intermediate care is the reduction of **delayed transfers of care** from hospitals. When intermediate care beds are accessible quickly, hospital patients can be discharged promptly to the appropriate facility, freeing hospital beds for acute patients. Our DES model indirectly captures this by noting that shorter waits mean fewer patients staying extra days in hospital. In scenario analyses with 24/7 admission and fast triage, we expect the number of patients experiencing a delay >1 day to drop significantly, thereby decreasing “bed-blocking” in hospitals. This has cost benefits and improves patient outcomes (since prolonged hospital stays can lead to complications. The simulation thus provides evidence for broader system benefits: improved intermediate care availability helps the entire healthcare continuum.

In summary, the DES model serves as a **decision-support tool** for intermediate care planning in the Arnhem and Nijmegen region. It allows stakeholders to virtually test policy changes – such as around-the-clock admissions, streamlined triage, or capacity increases – and see the projected effects on bed availability, occupancy, and waiting times. The results underscore that focusing on **operational efficiency (triage and admission logistics)** can unlock existing capacity and greatly reduce waits, while strategic increases in capacity can further ensure that surges in demand are handled without delays. By implementing the insights from this simulation, healthcare providers and administrators can optimize the intermediate care system, minimizing unnecessary hospital stays and ensuring that elderly patients receive timely, appropriate care on their road to recovery.

## References

ActiZ. (n.d.). _Hoe staat het met: de wachtlijsten voor verpleeghuizen?_ Retrieved January 15, 2024, from **https://www.actiz.nl/hoe-staat-het-met-de-wachtlijsten-voor-verpleeghuizen**

Arntzen, R. J., Bekker, R., & van der Mei, R. D. (2024). _Preference-based allocation of patients to nursing homes_ [Preprint]. SSRN. https://doi.org/10.2139/ssrn.4670165

Arntzen, R. J., Bekker, R., Smeekes, O. S., Buurman, B. M., Willems, H. C., Bhulai, S., & van der Mei, R. D. (2022). _Reduced waiting times by preference-based allocation of patients to nursing homes_. **Journal of the American Medical Directors Association, 23**(12), 2010–2014.e1. https://doi.org/10.1016/j.jamda.2022.04.012

Arntzen, R. J., van den Besselaar, J. H., Bekker, R., Buurman, B. M., & van der Mei, R. D. (2023). _Avoiding hospital admissions and delayed transfers of care by improved access to intermediate care: A simulation study_. **Journal of the American Medical Directors Association, 24**(7), 945–950.e4. https://doi.org/10.1016/j.jamda.2023.04.026

Bekker, R., Koole, G., & Roubos, D. (2017). _Flexible bed allocations for hospital wards_. **Health Care Management Science, 20**(4), 453–466. https://doi.org/10.1007/s10729-016-9364-4

Centrum Wiskunde & Informatica. (n.d.). _Halvering wachtlijsten in de ouderenzorg mogelijk door nieuw wiskundig model_. Retrieved January 15, 2024, from **https://www.cwi.nl/nl/samenwerkingen/showcases/cwi-sigra-and-amsterdam-umc-join-forces-to-improve-elderly-care/**

de Groot, A. J., Wattel, E. M., van Balen, R., Hertogh, C. M. P. M., & van der Wouden, J. C. (2023). _Discharge to rehabilitation-oriented care after acute hospital stay: Association with vulnerability screening on hospital admission_. **Annals of Geriatric Medicine and Research, 27**(4), 301–309. https://doi.org/10.4235/agmr.23.0068

Melis, R. J. F., Olde Rikkert, M. G. M., Parker, S. G., & van Eijken, M. I. J. (2004). _What is intermediate care?_ **BMJ, 329**(7462), 360–361. https://doi.org/10.1136/bmj.329.7462.360

Steiner, A. (2001). _Intermediate care—a good thing?_ **Age and Ageing, 30**(Suppl. 3), 33–39. https://doi.org/10.1093/ageing/30.suppl_3.33

Thijssen, R. (2023, May 15). _Dolce Vita: datagedreven optimalisatie voor de ouderenzorg_. Actie Leer Netwerk. **https://www.actieleernetwerk.nl/dolce-vita-datagedreven-optimalisatie-ouderenzorg/**

van Loon, C. G. (2024). _Reducing waiting times in acute elderly care: A case study for the region of Twente_ [Master’s thesis, Vrije Universiteit Amsterdam]. Amsterdam, The Netherlands.
