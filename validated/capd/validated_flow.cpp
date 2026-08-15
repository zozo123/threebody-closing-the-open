#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "capd/capdlib.h"

using namespace capd;
using capd::autodiff::Node;

namespace {

struct Seed {
  std::string name;
  double m1, m2, m3;
  double x1, v1, v2, period;
};

struct FloquetIntervals {
  interval alpha, beta, discriminant, plusOne, minusOne;
};

std::vector<std::string> splitTabs(const std::string& line) {
  std::vector<std::string> out;
  std::stringstream ss(line);
  std::string item;
  while (std::getline(ss, item, '\t')) out.push_back(item);
  return out;
}

Seed readSeed(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("cannot open seed file: " + path);
  std::string headerLine, rowLine, line;
  while (std::getline(in, line)) {
    if (line.empty() || line[0] == '#') continue;
    if (headerLine.empty()) headerLine = line;
    else { rowLine = line; break; }
  }
  if (headerLine.empty() || rowLine.empty()) throw std::runtime_error("seed TSV needs header and one row");
  auto h = splitTabs(headerLine);
  auto v = splitTabs(rowLine);
  if (h.size() != v.size()) throw std::runtime_error("seed TSV column mismatch");
  std::map<std::string,std::string> d;
  for (std::size_t i=0;i<h.size();++i) d[h[i]]=v[i];
  auto req = [&](const std::string& k)->std::string {
    auto it=d.find(k); if(it==d.end()) throw std::runtime_error("missing seed column: "+k); return it->second;
  };
  return Seed{req("name"), std::stod(req("m1")), std::stod(req("m2")), std::stod(req("m3")),
              std::stod(req("x1")), std::stod(req("v1")), std::stod(req("v2")), std::stod(req("period"))};
}

// Translation-reduced 8D vector field in
// (r1-r3, r2-r3, v1-v3, v2-v3). This implementation is independent of the
// Python and Julia ATLAS dynamics. CAPD automatic differentiation supplies the
// rigorous first variational enclosure.
void reducedField(Node /*t*/, Node in[], int /*dimIn*/, Node out[], int /*dimOut*/,
                  Node params[], int /*noParams*/) {
  Node m1=params[0], m2=params[1], m3=params[2];
  Node q1x=in[0], q1y=in[1], q2x=in[2], q2y=in[3];
  Node d12x=q2x-q1x, d12y=q2y-q1y;
  // Explicit products are used rather than x^2 because CAPD's autodiff parser
  // overloads operator^ for a scalar exponent, while an integer literal can be
  // promoted to a Node in this callback signature.
  Node r1sq=q1x*q1x + q1y*q1y;
  Node r2sq=q2x*q2x + q2y*q2y;
  Node r12sq=d12x*d12x + d12y*d12y;
  // -1.5 is exactly representable in binary; CAPD's official rigorous PCR3BP
  // example uses this same power convention for inverse-cube forces.
  Node inv1=r1sq^-1.5, inv2=r2sq^-1.5, inv12=r12sq^-1.5;
  Node g1x=q1x*inv1, g1y=q1y*inv1;
  Node g2x=q2x*inv2, g2y=q2y*inv2;
  Node g12x=d12x*inv12, g12y=d12y*inv12;

  out[0]=in[4]; out[1]=in[5]; out[2]=in[6]; out[3]=in[7];
  out[4]=m2*g12x-(m1+m3)*g1x-m2*g2x;
  out[5]=m2*g12y-(m1+m3)*g1y-m2*g2y;
  out[6]=-m1*g12x-m1*g1x-(m2+m3)*g2x;
  out[7]=-m1*g12y-m1*g1y-(m2+m3)*g2y;
}

IVector chartState(const Seed& s) {
  const double v3=-(s.m1*s.v1+s.m2*s.v2)/s.m3;
  IVector z(8);
  z[0]=s.x1; z[1]=0.; z[2]=1.; z[3]=0.;
  z[4]=0.; z[5]=s.v1-v3; z[6]=0.; z[7]=s.v2-v3;
  return z;
}

interval pairDistanceSquared(const IVector& z, int pair) {
  if(pair==0) return sqr(z[0])+sqr(z[1]);
  if(pair==1) return sqr(z[2])+sqr(z[3]);
  return sqr(z[2]-z[0])+sqr(z[3]-z[1]);
}

FloquetIntervals floquetIntervals(const IMatrix& M) {
  interval alpha=0., trM2=0.;
  for(int i=0;i<8;++i) {
    alpha += M[i][i];
    for(int j=0;j<8;++j) trM2 += M[i][j]*M[j][i];
  }
  interval beta=(sqr(alpha)-trM2)/2.;
  interval plusOne=beta-6.*alpha+20.;
  interval minusOne=beta-2.*alpha+4.;
  interval discriminant=sqr(alpha-4.)-4.*(beta-4.*alpha+8.);
  return FloquetIntervals{alpha,beta,discriminant,plusOne,minusOne};
}

bool finiteInterval(const interval& x) {
  return std::isfinite(x.leftBound()) && std::isfinite(x.rightBound());
}

std::string intervalJson(const interval& x) {
  std::ostringstream os;
  os << std::setprecision(17) << "[" << x.leftBound() << "," << x.rightBound() << "]";
  return os.str();
}

std::string vectorJson(const IVector& v) {
  std::ostringstream os; os << "[";
  for(int i=0;i<v.dimension();++i) { if(i) os << ","; os << intervalJson(v[i]); }
  os << "]"; return os.str();
}

std::string floquetJson(const FloquetIntervals& f) {
  std::ostringstream os;
  os << "{\"alpha\":" << intervalJson(f.alpha)
     << ",\"beta\":" << intervalJson(f.beta)
     << ",\"discriminant\":" << intervalJson(f.discriminant)
     << ",\"plus_one_event\":" << intervalJson(f.plusOne)
     << ",\"minus_one_event\":" << intervalJson(f.minusOne) << "}";
  return os.str();
}

} // namespace

int main(int argc, char** argv) {
  if(argc < 3 || argc > 4) {
    std::cerr << "usage: validated_flow SEED_TSV OUTPUT_JSON [BOX_RADIUS]\n";
    return 2;
  }
  const std::string seedPath=argv[1], outputPath=argv[2];
  const double boxRadius=(argc==4 ? std::stod(argv[3]) : 1e-13);
  if(!(boxRadius>=0. && std::isfinite(boxRadius))) {
    std::cerr << "invalid box radius\n"; return 2;
  }

  try {
    const Seed s=readSeed(seedPath);
    IMap vf(reducedField,8,8,3);
    vf.setParameter(0,interval(s.m1));
    vf.setParameter(1,interval(s.m2));
    vf.setParameter(2,interval(s.m3));
    IOdeSolver solver(vf,30);
    ITimeMap timeMap(solver);

    const IVector center=chartState(s);
    C1HORect2Set pointSet(center);
    const IVector finalPoint=timeMap(interval(s.period),pointSet);
    const IMatrix monodromy=(IMatrix)pointSet;
    const FloquetIntervals pointFloquet=floquetIntervals(monodromy);

    IVector box=center;
    for(int i=0;i<box.dimension();++i) {
      const double c=center[i].leftBound();
      const double r=boxRadius*std::max(1.0,std::abs(c));
      box[i]=interval(c-r,c+r);
    }
    C1HORect2Set boxSet(box);
    const IVector finalBox=timeMap(interval(s.period),boxSet);
    const IMatrix boxMonodromy=(IMatrix)boxSet;
    const FloquetIntervals boxFloquet=floquetIntervals(boxMonodromy);

    IVector closure(8);
    double maxClosureWidth=0., maxMonodromyWidth=0., maxBoxMonodromyWidth=0.;
    bool finite=true;
    for(int i=0;i<8;++i) {
      closure[i]=finalPoint[i]-center[i];
      finite = finite && finiteInterval(closure[i]);
      maxClosureWidth=std::max(maxClosureWidth,width(closure[i]));
      for(int j=0;j<8;++j) {
        finite=finite && finiteInterval(monodromy[i][j]) && finiteInterval(boxMonodromy[i][j]);
        maxMonodromyWidth=std::max(maxMonodromyWidth,width(monodromy[i][j]));
        maxBoxMonodromyWidth=std::max(maxBoxMonodromyWidth,width(boxMonodromy[i][j]));
      }
    }
    finite = finite
      && finiteInterval(pointFloquet.alpha) && finiteInterval(pointFloquet.beta)
      && finiteInterval(pointFloquet.discriminant) && finiteInterval(pointFloquet.plusOne)
      && finiteInterval(pointFloquet.minusOne) && finiteInterval(boxFloquet.alpha)
      && finiteInterval(boxFloquet.beta) && finiteInterval(boxFloquet.discriminant)
      && finiteInterval(boxFloquet.plusOne) && finiteInterval(boxFloquet.minusOne);

    interval pointD13=pairDistanceSquared(finalPoint,0);
    interval pointD23=pairDistanceSquared(finalPoint,1);
    interval pointD12=pairDistanceSquared(finalPoint,2);
    interval boxD13=pairDistanceSquared(finalBox,0);
    interval boxD23=pairDistanceSquared(finalBox,1);
    interval boxD12=pairDistanceSquared(finalBox,2);
    const bool collisionFreeFinal = pointD13.leftBound()>0. && pointD23.leftBound()>0. && pointD12.leftBound()>0.;
    const bool boxCollisionFreeFinal = boxD13.leftBound()>0. && boxD23.leftBound()>0. && boxD12.leftBound()>0.;
    const bool passed = finite && collisionFreeFinal && boxCollisionFreeFinal;

    std::ofstream out(outputPath);
    if(!out) throw std::runtime_error("cannot open output JSON");
    out << std::setprecision(17)
        << "{\n"
        << "  \"claim_status\": \"validated_flow_scaffolding\",\n"
        << "  \"proof_scope\": \"rigorous CAPD full-period flow, C1 variational and trace-invariant enclosure only; not a periodic-orbit existence proof and not an organizer certificate\",\n"
        << "  \"seed_name\": \"" << s.name << "\",\n"
        << "  \"masses\": [" << s.m1 << "," << s.m2 << "," << s.m3 << "],\n"
        << "  \"period\": " << s.period << ",\n"
        << "  \"input_box_relative_radius\": " << boxRadius << ",\n"
        << "  \"point_flow_final\": " << vectorJson(finalPoint) << ",\n"
        << "  \"point_flow_closure\": " << vectorJson(closure) << ",\n"
        << "  \"point_floquet_intervals\": " << floquetJson(pointFloquet) << ",\n"
        << "  \"box_floquet_intervals\": " << floquetJson(boxFloquet) << ",\n"
        << "  \"max_point_closure_interval_width\": " << maxClosureWidth << ",\n"
        << "  \"max_point_monodromy_interval_width\": " << maxMonodromyWidth << ",\n"
        << "  \"max_box_monodromy_interval_width\": " << maxBoxMonodromyWidth << ",\n"
        << "  \"final_pair_distance_squared_point\": [" << intervalJson(pointD13) << "," << intervalJson(pointD23) << "," << intervalJson(pointD12) << "],\n"
        << "  \"final_pair_distance_squared_box\": [" << intervalJson(boxD13) << "," << intervalJson(boxD23) << "," << intervalJson(boxD12) << "],\n"
        << "  \"finite_enclosure\": " << (finite?"true":"false") << ",\n"
        << "  \"collision_excluded_at_final_point_enclosure\": " << (collisionFreeFinal?"true":"false") << ",\n"
        << "  \"collision_excluded_at_final_box_enclosure\": " << (boxCollisionFreeFinal?"true":"false") << ",\n"
        << "  \"passed\": " << (passed?"true":"false") << "\n"
        << "}\n";

    std::cout << "seed=" << s.name
              << " validated_flow=" << (passed?"PASS":"FAIL")
              << " closure_width=" << maxClosureWidth
              << " point_monodromy_width=" << maxMonodromyWidth
              << " box_monodromy_width=" << maxBoxMonodromyWidth
              << " G+=" << pointFloquet.plusOne
              << " G-=" << pointFloquet.minusOne << "\n";
    return passed ? 0 : 1;
  } catch(const std::exception& e) {
    std::cerr << "CAPD validated-flow failure: " << e.what() << "\n";
    return 1;
  }
}
