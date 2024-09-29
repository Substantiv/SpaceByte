#include "ros/ros.h"

#include <iostream>
#include <vector>
#include <casadi/casadi.hpp>

using namespace std;
using namespace casadi;

void timerCallBack(const ros::TimerEvent& event){
    ROS_INFO("CasADi Test !!!");

    // create optimization variables
    SX x = SX::sym("x", 3);

    // create parameter
    SX p = SX::sym("p", 2);

    // Objective
    SX f = x(0) * x(0) + x(1) * x(1) + x(2) * x(2);

    // Constrains
    SX g = vertcat(6 * x(0) + 3 * x(1) + 2 * x(2) - p(0), p(1) * x(0) + x(1) - x(2) - 1);

    // Initial guess and bounds for the optimization variables
    vector<double> x0 = { 0.15, 0.15, 0.00 };
    vector<double> lbx = { 0.00, 0.00, 0.00 };
    vector<double> ubx = { inf, inf, inf };

    // Nonlinear bounds
    vector<double> lbg = { 0.00, 0.00 };
    vector<double> ubg = { 0.00, 0.00 };

    // Original parameter values
    vector<double> p0 = { 5.00, 1.00 };

    // NLP
    SXDict nlp = { { "x", x }, { "p", p }, { "f", f }, { "g", g } };

    // Create NLP solver and buffers
    Function solver = nlpsol("solver", "ipopt", nlp);
    std::map<std::string, DM> arg, res;

    // Solve the NLP
    arg["lbx"] = lbx;
    arg["ubx"] = ubx;
    arg["lbg"] = lbg;
    arg["ubg"] = ubg;
    arg["x0"] = x0;
    arg["p"] = p0;
    res = solver(arg);
}

int main(int argc, char *argv[]){
    ros::init(argc, argv, "timer_use");
    ros::NodeHandle nh;
    ros::Timer timer = nh.createTimer(ros::Duration(1), timerCallBack);

    ros::spin();
    return 0;
}

