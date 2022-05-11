# Test rig anomaly detection and condition monitoring for predictive maintenance 

The repository contains the development of applications to detect anomalies in test results (anomaly detection), to monitor the test rig condition (forecasting), and to predict a need for maintenance in advance of a rig failure or excessive deterioration of its performance. The code also serves to provide advanced analytics of test article performance over time or during the test cycle.

# Application serving

Both applications (anomaly detector and forecaster) follow the same cloud architecture to continuously build, deploy and serve the app.

![App serving architecture](https://github.com/ivanokhotnikov/test_rig/blob/master/images/serving_architecture.png?raw=True)


# Scripts

The directory contains the utility module `util` to pre- and post-process (see `readers.py`), to visualize (see `plotters.py`) the development results, to save and load preliminary output models, images, etc. The configurations are stored in  the `config.py` governing features sets, hyper-parameters tuning and directories.

# Links

[Anomaly detector](https://anomaly-detector.hydreco.uk/)

[Forecasting](https://forecasting.hydreco.uk/)

# Usage

```
streamlit run anomaly_detection\run_detector.py
streamlit run forecasting\run_forecaster.py
```