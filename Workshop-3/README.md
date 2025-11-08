# Workshop 3 — Robust System Design and Project Management

*Kaggle competition:* [Data Mining Hackathon on BIG DATA (7GB) — Best Buy Mobile Web Site](https://www.kaggle.com/competitions/acm-sf-chapter-hackathon-big/overview)

*Workshop Report:* Workshop 3

---

## System Design and Management Summary

We refined the system architecture to improve scalability, reliability, and maintainability. Key updates include a load balancer for horizontal scaling, database replication for fault tolerance, and a queueing mechanism to ensure asynchronous, reliable communication between application and ML services. A monitoring layer (e.g., Prometheus or ELK Stack) was proposed to track system and model metrics. The design follows quality frameworks (ISO 9001, CMMI, Six Sigma) to support continuous improvement and reduce variability.

A detailed risk analysis for the Click2Buy system was produced, addressing model drift, data integrity issues, pipeline dependency failures, and scalability bottlenecks. For each risk we defined severity, mitigation strategies (retraining, schema enforcement, checkpointing, circuit breakers), and monitoring/alerting procedures to maintain system robustness and rapid response.

For project management we adopted Scrum with Kanban tracking. Roles were defined (Product Owner, Scrum Master, Data/System Analyst, Developer/ML Engineer) and milestones set for Implementation, Simulation & Validation, Evaluation, and Final Documentation. This workshop consolidates the evolution from systems analysis (Workshop 1) and systems design (Workshop 2) toward a robust, production-ready architecture.

---

## Additional Sources

- *Workshop Report 1:* [Workshop 1](https://github.com/MelisaMelenge/Data-Mining-Hackathon-on-BIG-DATA-7GB-Best-Buy-mobile-web-site/blob/main/Workshop%201/Workshop1.pdf)
- *Workshop Report 2:* [Workshop 2](https://github.com/MelisaMelenge/Data-Mining-Hackathon-on-BIG-DATA-7GB-Best-Buy-mobile-web-site/blob/main/Workshop_2_Design/Workshop2.pdf)
- *Catch up Report :* [Report](https://github.com/MelisaMelenge/Data-Mining-Hackathon-on-BIG-DATA-7GB-Best-Buy-mobile-web-site/blob/main/Project%20Catch-Up/Click2Buy__Predicting_User_Interest_on_BestBuy_s_Mobile_Platform%20(Report).pdf)
- Draw.io: https://app.diagrams.net/  
- Best Buy Official Site: https://www.bestbuy.com/
- Canva: https://www.canva.com/es_419/

---

## Authors

- Melisa Maldonado Melenge  
- Jean Pierre Mora Cepeda  
- Juan Diego Martínez Beltrán  
- Luis Felipe Suárez Sánchez
