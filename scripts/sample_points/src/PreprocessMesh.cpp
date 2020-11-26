// Copyright 2004-2019 Facebook. All Rights Reserved.
// Copyright 2020 Korrawe Karunratanakul. All Rights Reserved.

#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <pangolin/geometry/geometry.h>
#include <pangolin/geometry/glgeometry.h>
#include <pangolin/gl/gl.h>
#include <pangolin/pangolin.h>

#include <CLI/CLI.hpp>
#include <cnpy.h>

#include "Utils.h"

extern pangolin::GlSlProgram GetShaderProgram();

void SampleFromSurface(pangolin::Geometry &geom,
                        std::vector<Eigen::Vector3f> &surfpts,
                        std::vector<int> &tri_indices,
                        int num_sample) {
  float total_area = 0.0f;

  std::vector<float> cdf_by_area;

  std::vector<Eigen::Vector3i> linearized_faces;

  for (const auto &object : geom.objects) {
    auto it_vert_indices = object.second.attributes.find("vertex_indices");
    if (it_vert_indices != object.second.attributes.end()) {

      pangolin::Image<uint32_t> ibo =
          pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);
        
      std::cout << ibo.h << " " << ibo.w << std::endl;

      for (int i = 0; i < ibo.h; ++i) {
        linearized_faces.emplace_back(ibo(0, i), ibo(1, i), ibo(2, i));
      }
    }
  }

  pangolin::Image<float> vertices = pangolin::get<pangolin::Image<float>>(
      geom.buffers["geometry"].attributes["vertex"]);

  for (const Eigen::Vector3i &face : linearized_faces) {

    float area = TriangleArea(
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        (Eigen::Vector3f)Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2))));

    if (std::isnan(area)) {
      area = 0.f;
    }

    total_area += area;

    if (cdf_by_area.empty()) {

      cdf_by_area.push_back(area);

    } else {

      cdf_by_area.push_back(cdf_by_area.back() + area);
    }
  }

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, total_area);

  while ((int)surfpts.size() < num_sample) {
    float tri_sample = rand_dist(generator);
    std::vector<float>::iterator tri_index_iter =
        lower_bound(cdf_by_area.begin(), cdf_by_area.end(), tri_sample);
    int tri_index = tri_index_iter - cdf_by_area.begin();

    const Eigen::Vector3i &face = linearized_faces[tri_index];
    tri_indices.push_back(tri_index);

    surfpts.push_back(SamplePointFromTriangle(
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(0))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(1))),
        Eigen::Map<Eigen::Vector3f>(vertices.RowPtr(face(2)))));
  }
}


void SampleSDFNearSurface(KdVertexListTree &kdTree,
                           std::vector<Eigen::Vector3f> &vertices,
                           std::vector<std::size_t> &point_tri_ids,
                           std::vector<Eigen::Vector3f> &xyz_surf,
                           std::vector<Eigen::Vector3f> &normals,
                           std::vector<Eigen::Vector3f> &xyz,
                           std::vector<float> &sdfs, 
                           std::vector<int> &tri_indices, /// 
                           std::vector<int> &sdf_tri_indices, ///
                           int num_rand_samples,
                           float variance, float second_variance,
                           float bounding_cube_dim, int num_votes) {
  float stdv = sqrt(variance);

  std::random_device seeder;
  std::mt19937 generator(seeder());
  std::uniform_real_distribution<float> rand_dist(0.0, 1.0);
  std::vector<Eigen::Vector3f> xyz_used;
  std::vector<Eigen::Vector3f> second_samples;
  std::vector<int> sdf_tri_indices_used;

  std::random_device rd;
  std::mt19937 rng(rd());
  std::uniform_int_distribution<int> vert_ind(0, vertices.size() - 1);
  std::normal_distribution<float> perterb_norm(0, stdv);
  std::normal_distribution<float> perterb_second(0, sqrt(second_variance));

  for (unsigned int i = 0; i < xyz_surf.size(); i++) {
    Eigen::Vector3f surface_p = xyz_surf[i];
    Eigen::Vector3f samp1 = surface_p;
    Eigen::Vector3f samp2 = surface_p;

    for (int j = 0; j < 3; j++) {
      samp1[j] += perterb_norm(rng);
      samp2[j] += perterb_second(rng);
    }

    xyz.push_back(samp1);
    xyz.push_back(samp2);
    sdf_tri_indices.push_back(tri_indices[i]);
    sdf_tri_indices.push_back(tri_indices[i]);
  }

  for (int s = 0; s < (int)(num_rand_samples); s++) {
    xyz.push_back(Eigen::Vector3f(
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2,
        rand_dist(generator) * bounding_cube_dim - bounding_cube_dim / 2));
    sdf_tri_indices.push_back(-1);
  }

  // now compute sdf for each xyz sample
  for (int s = 0; s < (int)xyz.size(); s++) {
    Eigen::Vector3f samp_vert = xyz[s];
    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);
    kdTree.knnSearch(samp_vert.data(), num_votes, cl_indices.data(),
                     cl_distances.data());

    int num_pos = 0;
    float sdf;
    uint32_t first_point_ind = -1;

    for (int ind = 0; ind < num_votes; ind++) {
      uint32_t cl_ind = cl_indices[ind];
      Eigen::Vector3f cl_vert = vertices[cl_ind];
      Eigen::Vector3f ray_vec = samp_vert - cl_vert;
      float ray_vec_leng = ray_vec.norm();

      if (ind == 0) {
        ///
        first_point_ind = cl_ind;
        // std::cout << ".. " <<  point_tri_ids[cl_ind] << std::endl;

        // if close to the surface, use point plane distance
        if (ray_vec_leng < stdv)
          sdf = fabs(normals[cl_ind].dot(ray_vec));
        else
          sdf = ray_vec_leng;
      }

      float d = normals[cl_ind].dot(ray_vec / ray_vec_leng);
      if (d > 0)
        num_pos++;
    }

    // all or nothing , else ignore the point
    if ((num_pos == 0) || (num_pos == num_votes)) {
      xyz_used.push_back(samp_vert);
      if (num_pos <= (num_votes / 2)) {
        sdf = -sdf;
      }
      sdfs.push_back(sdf);
      if (sdf_tri_indices[s] == -1)
        sdf_tri_indices_used.push_back(point_tri_ids[first_point_ind]); ///
      else
        sdf_tri_indices_used.push_back(sdf_tri_indices[s]); ///
    }
  }

  xyz = xyz_used;
  sdf_tri_indices = sdf_tri_indices_used; ///
}

void writeSDFToNPY(std::vector<Eigen::Vector3f> &xyz, std::vector<float> &sdfs,
                   std::string filename) {
  unsigned int num_vert = xyz.size();
  std::vector<float> data(num_vert * 4);
  int data_i = 0;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    for (int j = 0; j < 3; j++)
      data[data_i++] = v[j];
    data[data_i++] = s;
  }

  cnpy::npy_save(filename, &data[0], {(long unsigned int)num_vert, 4}, "w");
}

void writeSDFToNPZ(std::vector<Eigen::Vector3f> &xyz, std::vector<float> &sdfs,
                   std::vector<int> &tri_labels,
                   std::vector<float> &sdfs_to_other,
                   std::vector<int> &tri_labels_to_other,
                   std::string filename, bool print_num = false) {
  unsigned int num_vert = xyz.size();
  std::vector<float> pos;
  std::vector<float> neg;
  std::vector<int> labels_neg;
  std::vector<int> labels_pos;
  // for other
  std::vector<float> pos_other;
  std::vector<float> neg_other;
  std::vector<int> labels_neg_other;
  std::vector<int> labels_pos_other;

  for (unsigned int i = 0; i < num_vert; i++) {
    Eigen::Vector3f v = xyz[i];
    float s = sdfs[i];

    if (s > 0) {
      for (int j = 0; j < 3; j++)
        pos.push_back(v[j]);
      pos.push_back(s);
      pos_other.push_back(sdfs_to_other[i]); // other
      // label
      for (int j = 0; j < 4; j++) {
        // labels_pos.push_back(v[j]);
        labels_pos.push_back(tri_labels[i*4 + j]);
        labels_pos_other.push_back(tri_labels_to_other[i*4 + j]);
      }

    } else {
      for (int j = 0; j < 3; j++)
        neg.push_back(v[j]);
      neg.push_back(s);
      neg_other.push_back(sdfs_to_other[i]); // other
      // label
      for (int j = 0; j < 4; j++) {
        // labels_neg.push_back(v[j]);
        labels_neg.push_back(tri_labels[i*4 + j]);
        labels_neg_other.push_back(tri_labels_to_other[i*4 + j]);
      }
    }
  }


  cnpy::npz_save(filename, "pos", &pos[0],
                 {(long unsigned int)(pos.size() / 4.0), 4}, "w");
  cnpy::npz_save(filename, "neg", &neg[0],
                 {(long unsigned int)(neg.size() / 4.0), 4}, "a");
  cnpy::npz_save(filename, "lab_pos", &labels_pos[0],
                 {(long unsigned int)(labels_pos.size() / 4.0), 4}, "a");
  cnpy::npz_save(filename, "lab_neg", &labels_neg[0],
                 {(long unsigned int)(labels_neg.size() / 4.0), 4}, "a");
  // to another mesh
  cnpy::npz_save(filename, "pos_other", &pos_other[0],
                 {(long unsigned int)(pos_other.size()), 1}, "a");
  cnpy::npz_save(filename, "neg_other", &neg_other[0],
                 {(long unsigned int)(neg_other.size()), 1}, "a");
  cnpy::npz_save(filename, "lab_pos_other", &labels_pos_other[0],
                 {(long unsigned int)(labels_pos_other.size() / 4.0), 4}, "a");
  cnpy::npz_save(filename, "lab_neg_other", &labels_neg_other[0],
                 {(long unsigned int)(labels_neg_other.size() / 4.0), 4}, "a");

  if (print_num) {
    std::cout << "pos num: " << pos.size() / 4.0 << std::endl;
    std::cout << "neg num: " << neg.size() / 4.0 << std::endl;
  }
}

void writeSDFToPLY(std::vector<Eigen::Vector3f> &xyz, std::vector<float> &sdfs,
                   std::string filename, bool neg_only = true,
                   bool pos_only = false) {
  int num_verts;
  if (neg_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s <= 0)
        num_verts++;
    }
  } else if (pos_only) {
    num_verts = 0;
    for (int i = 0; i < (int)sdfs.size(); i++) {
      float s = sdfs[i];
      if (s >= 0)
        num_verts++;
    }
  } else {
    num_verts = xyz.size();
  }

  std::ofstream plyFile;
  plyFile.open(filename);
  plyFile << "ply\n";
  plyFile << "format ascii 1.0\n";
  plyFile << "element vertex " << num_verts << "\n";
  plyFile << "property float x\n";
  plyFile << "property float y\n";
  plyFile << "property float z\n";
  plyFile << "property uchar red\n";
  plyFile << "property uchar green\n";
  plyFile << "property uchar blue\n";
  plyFile << "end_header\n";

  for (int i = 0; i < (int)sdfs.size(); i++) {
    Eigen::Vector3f v = xyz[i];
    float sdf = sdfs[i];
    bool neg = (sdf <= 0);
    bool pos = (sdf >= 0);
    if (neg)
      sdf = -sdf;
    int sdf_i = std::min((int)(sdf * 255), 255);
    if (!neg_only && pos)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << 0 << " " << 0
              << " " << sdf_i << "\n";
    if (!pos_only && neg)
      plyFile << v[0] << " " << v[1] << " " << v[2] << " " << sdf_i << " " << 0
              << " " << 0 << "\n";
  }
  plyFile.close();
}

// Copied from sampleMeshSurface
void SaveNormalizationParamsToNPZ(
    const Eigen::Vector3f offset,
    const float scale,
    const std::string filename) {
  cnpy::npz_save(filename, "offset", offset.data(), {3ul}, "w");
  cnpy::npz_save(filename, "scale", &scale, {1ul}, "a");
}

//////
void preprocessMesh(pangolin::Geometry &geom) {
  // linearize the object indices
  {
    int total_num_faces = 0;

    for (const auto &object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {

        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        total_num_faces += ibo.h;
      }
    }

    //      const int total_num_indices = total_num_faces * 3;
    pangolin::ManagedImage<uint8_t> new_buffer(3 * sizeof(uint32_t),
                                               total_num_faces);

    pangolin::Image<uint32_t> new_ibo =
        new_buffer.UnsafeReinterpret<uint32_t>().SubImage(0, 0, 3,
                                                          total_num_faces);

    int index = 0;

    for (const auto &object : geom.objects) {
      auto it_vert_indices = object.second.attributes.find("vertex_indices");
      if (it_vert_indices != object.second.attributes.end()) {

        pangolin::Image<uint32_t> ibo =
            pangolin::get<pangolin::Image<uint32_t>>(it_vert_indices->second);

        std::cout << "ibo.h " << ibo.h << std::endl;
        for (int i = 0; i < ibo.h; ++i) {
          
          // std::cout << ibo(i,0) << " " << ibo(i,1) << " " << ibo(i,2) << std::endl;
          // std::cout << ibo.Row(i) std::endl;
          new_ibo.Row(index).CopyFrom(ibo.Row(i));
          ++index;
        }
      }
    }

    geom.objects.clear();
    auto faces = geom.objects.emplace(std::string("mesh"),
                                      pangolin::Geometry::Element());

    faces->second.Reinitialise(3 * sizeof(uint32_t), total_num_faces);

    faces->second.CopyFrom(new_buffer);

    new_ibo = faces->second.UnsafeReinterpret<uint32_t>().SubImage(
        0, 0, 3, total_num_faces);
    faces->second.attributes["vertex_indices"] = new_ibo;
  }
  // remove textures
  geom.textures.clear();
}

void validate_point(pangolin::Geometry &geom,
                  pangolin::Image<uint32_t> modelFaces,
                  bool vis,
                  float max_dist,
                  float rejection_criteria_obs,
                  float rejection_criteria_tri,
                  std::vector<Eigen::Vector3f> &vertices_out,
                  std::vector<Eigen::Vector3f> &normals_out,
                  std::vector<std::size_t> &point_tri_ids_out,
                  bool isObject) {
  
  std::string windowName;
  if (isObject) {
    windowName = "Main_Obj";
  } else {
    windowName = "Main_Hand";
  }
  if (vis)
    pangolin::CreateWindowAndBind(windowName, 640, 480);
  else
    pangolin::CreateWindowAndBind(windowName, 1, 1);
  glEnable(GL_DEPTH_TEST);
  glDisable(GL_DITHER);
  glDisable(GL_POINT_SMOOTH);
  glDisable(GL_LINE_SMOOTH);
  glDisable(GL_POLYGON_SMOOTH);
  glHint(GL_POINT_SMOOTH, GL_DONT_CARE);
  glHint(GL_LINE_SMOOTH, GL_DONT_CARE);
  glHint(GL_POLYGON_SMOOTH_HINT, GL_DONT_CARE);
  glDisable(GL_MULTISAMPLE_ARB);
  glShadeModel(GL_FLAT);
  
  // Define Projection and initial ModelView matrix
  pangolin::OpenGlRenderState s_cam(
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, -max_dist,
                                             max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));
  pangolin::OpenGlRenderState s_cam2(
      pangolin::ProjectionMatrixOrthographic(-max_dist, max_dist, max_dist,
                                             -max_dist, 0, 2.5),
      pangolin::ModelViewLookAt(0, 0, -1, 0, 0, 0, pangolin::AxisY));

  // Create Interactive View in window
  pangolin::Handler3D handler(s_cam);

  pangolin::GlGeometry gl_geom = pangolin::ToGlGeometry(geom);

  pangolin::GlSlProgram prog = GetShaderProgram();

  if (vis) {
    pangolin::View &d_cam = pangolin::CreateDisplay()
                                .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f / 480.0f)
                                .SetHandler(&handler);

    while (!pangolin::ShouldQuit()) {
      // Clear screen and activate view to render into
      glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

      d_cam.Activate(s_cam);

      prog.Bind();
      prog.SetUniform("MVP", s_cam.GetProjectionModelViewMatrix());
      prog.SetUniform("V", s_cam.GetModelViewMatrix());

      pangolin::GlDraw(prog, gl_geom, nullptr);
      prog.Unbind();

      // Swap frames and Process Events
      pangolin::FinishFrame();
    }
  }
  // Create Framebuffer with attached textures
  size_t w = 400;
  size_t h = 400;
  pangolin::GlRenderBuffer zbuffer(w, h, GL_DEPTH_COMPONENT32);
  pangolin::GlTexture normals(w, h, GL_RGBA32F);
  pangolin::GlTexture vertices(w, h, GL_RGBA32F);
  pangolin::GlFramebuffer framebuffer(vertices, normals, zbuffer);

  // View points around a sphere.
  std::vector<Eigen::Vector3f> views =
      EquiDistPointsOnSphere(100, max_dist * 1.1);

  std::vector<Eigen::Vector4f> point_normals;
  std::vector<Eigen::Vector4f> point_verts;
  std::vector<std::size_t> point_tri_ids; ///
  size_t num_tri = modelFaces.h;
  std::vector<Eigen::Vector4f> tri_id_normal_test(num_tri);
  for (size_t j = 0; j < num_tri; j++)
    tri_id_normal_test[j] = Eigen::Vector4f(0.0f, 0.0f, 0.0f, 0.0f);
  int total_obs = 0;
  int wrong_obs = 0;
  for (unsigned int v = 0; v < views.size(); v++) {
    // change camera location
    s_cam2.SetModelViewMatrix(pangolin::ModelViewLookAt(
        views[v][0], views[v][1], views[v][2], 0, 0, 0, pangolin::AxisY));
    // Draw the scene to the framebuffer
    framebuffer.Bind();
    glViewport(0, 0, w, h);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
    prog.Bind();
    prog.SetUniform("MVP", s_cam2.GetProjectionModelViewMatrix());
    prog.SetUniform("V", s_cam2.GetModelViewMatrix());
    prog.SetUniform("ToWorld", s_cam2.GetModelViewMatrix().Inverse());
    prog.SetUniform("slant_thr", -1.0f, 1.0f);
    prog.SetUniform("ttt", 1.0, 0, 0, 1);
    pangolin::GlDraw(prog, gl_geom, nullptr);
    prog.Unbind();

    framebuffer.Unbind();
    pangolin::TypedImage img_normals;
    normals.Download(img_normals);
    std::vector<Eigen::Vector4f> im_norms = ValidPointsAndTrisFromIm(
        img_normals.UnsafeReinterpret<Eigen::Vector4f>(), tri_id_normal_test,
        point_tri_ids, 
        total_obs, wrong_obs);
    point_normals.insert(point_normals.end(), im_norms.begin(), im_norms.end());
    pangolin::TypedImage img_verts;
    vertices.Download(img_verts);
    std::vector<Eigen::Vector4f> im_verts =
        ValidPointsFromIm(img_verts.UnsafeReinterpret<Eigen::Vector4f>());
    point_verts.insert(point_verts.end(), im_verts.begin(), im_verts.end());
  }

  int bad_tri = 0;
  std::cout << "tri_id_normal_test.size() " << tri_id_normal_test.size() << std::endl;
  for (unsigned int t = 0; t < tri_id_normal_test.size(); t++) {
    if (tri_id_normal_test[t][3] < 0.0f)
      bad_tri++;
  }

  // std::cout << meshFileName << std::endl;
  std::cout << (float)(wrong_obs) / float(total_obs) << std::endl;
  std::cout << (float)(bad_tri) / float(num_tri) << std::endl;

  float wrong_ratio = (float)(wrong_obs) / float(total_obs);
  float bad_tri_ratio = (float)(bad_tri) / float(num_tri);

  if (wrong_ratio > rejection_criteria_obs ||
      bad_tri_ratio > rejection_criteria_tri) {
    if (isObject) {
      std::cout << "mesh rejected object" << std::endl;
    } else {
      std::cout << "mesh rejected hand" << std::endl;
    }
  }

  std::vector<Eigen::Vector3f> vertices2;
  std::vector<Eigen::Vector3f> normals2;

  for (unsigned int v = 0; v < point_verts.size(); v++) {
    vertices2.push_back(point_verts[v].head<3>());
    normals2.push_back(point_normals[v].head<3>());
  }

  vertices_out = vertices2;
  normals_out = normals2;
  point_tri_ids_out = point_tri_ids;
}

void sample_point(KdVertexListTree &kdTree_surf,
                  pangolin::Geometry &geom,
                  int8_t * face_labels,
                  std::string npyFileName,
                  std::string plyFileNameOut,
                  int num_sample,
                  float variance, float second_variance,
                  std::vector<Eigen::Vector3f> &xyz_out, std::vector<float> &sdf_out, //
                  std::vector<int> &tri_labels_out, //
                  // KdVertexListTree &kdTree_out, //
                  std::vector<Eigen::Vector3f> &vertices2, //
                  std::vector<Eigen::Vector3f> &normals2, //
                  std::vector<std::size_t> &point_tri_ids, //
                  bool save_ply, bool isObject) {

  std::vector<Eigen::Vector3f> xyz;
  std::vector<Eigen::Vector3f> xyz_surf;
  std::vector<int> tri_indices; ///
  std::vector<int> tri_labels; ///
  std::vector<float> sdf;
  std::vector<int> sdf_tri_indices; ///
  int num_samp_near_surf = (int)(47 * num_sample / 50);
  std::cout << "num_samp_near_surf: " << num_samp_near_surf << std::endl;
  SampleFromSurface(geom, xyz_surf, tri_indices, num_samp_near_surf / 2);

  auto start = std::chrono::high_resolution_clock::now();
  SampleSDFNearSurface(kdTree_surf, vertices2, point_tri_ids,
                        xyz_surf, normals2, xyz, sdf, 
                        tri_indices, sdf_tri_indices,
                        num_sample - num_samp_near_surf, variance,
                        second_variance, 2, 11);

  for (int i=0; i<sdf_tri_indices.size(); i++) {
    if (isObject) {
      for (int j=0; j<4; j++)
        tri_labels.push_back(0);
      continue;
    }

    if (sdf_tri_indices[i] == -1)
      std::cout << "something is wrong here " << std::endl;
    else
      for (int j=0; j<4; j++)
        tri_labels.push_back(int(face_labels[sdf_tri_indices[i]*4 + j]));
  }
  auto finish = std::chrono::high_resolution_clock::now();
  auto elapsed =
      std::chrono::duration_cast<std::chrono::seconds>(finish - start).count();
  std::cout << elapsed << std::endl;

  if (save_ply) {

    writeSDFToPLY(xyz, sdf, plyFileNameOut, false, true);
  }

  std::cout << "num points sampled: " << xyz.size() << std::endl;
  std::size_t save_npz = npyFileName.find("npz");
  if (save_npz == std::string::npos)
    writeSDFToNPY(xyz, sdf, npyFileName);
  else {
    xyz_out = xyz;
    sdf_out = sdf;
    tri_labels_out = tri_labels;
  }
  if (isObject) {
      std::cout << "success object" << std::endl;
    } else {
      std::cout << "success hand" << std::endl;
    }

}


void calDistToOther(std::vector<Eigen::Vector3f> &xyz,
                    const KdVertexListTree &kdTree,
                    std::vector<Eigen::Vector3f> &vertices,
                    std::vector<Eigen::Vector3f> &normals,
                    std::vector<std::size_t> &point_tri_ids,
                    int8_t * face_labels,
                    std::vector<float> &sdf_out,
                    std::vector<int> &point_labels_out,
                    float variance,
                    bool isObject
                    ) {
  float stdv = sqrt(variance);

  std::vector<int> sdf_tri_indices; ///
  // start
  // use kdTree to calculate dist
  // now compute sdf for each xyz sample
  int num_votes = 10;
  for (int s = 0; s < (int)xyz.size(); s++) {
    Eigen::Vector3f samp_vert = xyz[s];
    std::vector<int> cl_indices(num_votes);
    std::vector<float> cl_distances(num_votes);

    kdTree.knnSearch(samp_vert.data(), num_votes, cl_indices.data(),
                     cl_distances.data());

    float sdf;
    uint32_t first_point_ind = -1;

    // use only the closest node
    uint32_t cl_ind = cl_indices[0];
    Eigen::Vector3f cl_vert = vertices[cl_ind];
    Eigen::Vector3f ray_vec = samp_vert - cl_vert;
    float ray_vec_leng = ray_vec.norm();

    // if close to the surface, use point plane distance
    if (ray_vec_leng < stdv)
      sdf = fabs(normals[cl_ind].dot(ray_vec));
    else
      sdf = ray_vec_leng;

    float d = normals[cl_ind].dot(ray_vec / ray_vec_leng);
    if (d <= 0) {
      sdf = -sdf;
    }
    sdf_out.push_back(sdf);
    sdf_tri_indices.push_back(point_tri_ids[cl_ind]);
  }
  
  for (int i=0; i<sdf_tri_indices.size(); i++) {
    if (isObject) {
      for (int j=0; j<4; j++)
        point_labels_out.push_back(0);
    }
    else 
      for (int j=0; j<4; j++){
        point_labels_out.push_back(int(face_labels[sdf_tri_indices[i]*4 + j]));
      }
  }
}


int main(int argc, char **argv) {
  std::string meshFileName;
  std::string objFileName;
  std::string normalizationOutputFile;
  bool vis = false;

  std::string npyFileName;
  std::string npyObjFileName;
  std::string plyFileNameOut;
  std::string plyObjFileNameOut;
  std::string spatial_samples_npz;
  bool save_ply = true;
  bool test_flag = false;
  float variance = 0.005;
  // int num_sample = 500000;
  int num_sample = 20000; // 10000;
  float rejection_criteria_obs = 0.02f;
  float rejection_criteria_tri = 0.03f;
  float num_samp_near_surf_ratio = 1.f; //47.0f / 50.0f;

  CLI::App app{"PreprocessMesh"};
  app.add_option("--hand", meshFileName, "Hand Mesh File Name for Reading")->required();
  app.add_option("--obj", objFileName, "Object Mesh File Name for Reading")->required(); //
  app.add_flag("-v", vis, "enable visualization");
  app.add_option("--outhand", npyFileName, "Save npy pc to here")->required();
  app.add_option("--outobj", npyObjFileName, "Save object npy pc to here")->required(); //
  app.add_option("--ply", plyFileNameOut, "Save ply pc to here");
  app.add_option("--plyobj", plyObjFileNameOut, "Save ply for obj pc to here");
  app.add_option("-s", num_sample, "Save ply pc to here");
  app.add_option("--var", variance, "Set Variance");
  app.add_flag("--sply", save_ply, "save ply point cloud for visualization");
  app.add_flag("-t", test_flag, "test_flag");
  app.add_option("--normalize", normalizationOutputFile, "Save normalization");
  app.add_option("-n", spatial_samples_npz, "spatial samples from file");

  CLI11_PARSE(app, argc, argv);

  if (test_flag)
    variance = 0.05;

  float second_variance = variance / 10;
  std::cout << "variance: " << variance << " second: " << second_variance
            << std::endl;
  if (test_flag) {
    second_variance = variance / 100;
    num_samp_near_surf_ratio = 45.0f / 50.0f;
    num_sample = 250000;
  }

  std::cout << spatial_samples_npz << std::endl;

  glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
  glPixelStorei(GL_UNPACK_ROW_LENGTH, 0);
  glPixelStorei(GL_UNPACK_SKIP_PIXELS, 0);
  glPixelStorei(GL_UNPACK_SKIP_ROWS, 0);

  // Hand
  pangolin::Geometry geom = pangolin::LoadGeometry(meshFileName);
  std::cout << geom.objects.size() << " hands" << std::endl;

  // Object
  pangolin::Geometry geomObj = pangolin::LoadGeometry(objFileName);
  std::cout << geomObj.objects.size() << " objects" << std::endl;

  //load into a new array
  cnpy::NpyArray arr = cnpy::npy_load("./scripts/sample_points/hand_model_seg/face2label_sealed.npy");
  int8_t * loaded_data = arr.data<int8_t>();
  size_t nrows = arr.shape[0];
  size_t ncols = arr.shape[1];
  std::cout << "nrow " << nrows << std::endl;
  std::cout << "ncols " << ncols << std::endl; 

  preprocessMesh(geom);
  preprocessMesh(geomObj);

  pangolin::Image<uint32_t> modelFaces =
      pangolin::get<pangolin::Image<uint32_t>>(
          geom.objects.begin()->second.attributes["vertex_indices"]);

  pangolin::Image<uint32_t> modelFacesObj =
      pangolin::get<pangolin::Image<uint32_t>>(
          geomObj.objects.begin()->second.attributes["vertex_indices"]);
  
  // Compute min and max values for hand
  const std::pair<Eigen::Vector3f, Eigen::Vector3f> minMaxPoint =
        ComputeMinMax(geom);

  // Compute min and max values for object
  const std::pair<Eigen::Vector3f, Eigen::Vector3f> minMaxPointObj =
        ComputeMinMax(geomObj);

  // Calculate center point of hand and object
  float xMin, yMin, zMin, xMax, yMax, zMax; 
  xMin = fmin(minMaxPoint.first[0], minMaxPointObj.first[0]);
  yMin = fmin(minMaxPoint.first[1], minMaxPointObj.first[1]);
  zMin = fmin(minMaxPoint.first[2], minMaxPointObj.first[2]);
  xMax = fmax(minMaxPoint.second[0], minMaxPointObj.second[0]);
  yMax = fmax(minMaxPoint.second[1], minMaxPointObj.second[1]);
  zMax = fmax(minMaxPoint.second[1], minMaxPointObj.second[2]);

  const Eigen::Vector3f center((xMax + xMin) / 2.0f, (yMax + yMin) / 2.0f,
                               (zMax + zMin) / 2.0f);

  std::cout << "min x" << minMaxPoint.first[0] << ", min y " << minMaxPoint.first[1] << ", min z " << minMaxPoint.first[2] << std::endl;
  std::cout << "min x" << minMaxPointObj.first[0] << ", min y " << minMaxPointObj.first[1] << ", min z " << minMaxPointObj.first[2] << std::endl;

  float max_dist_hand = ComputeMaxDistance(geom, center);
  float max_dist_obj = ComputeMaxDistance(geomObj, center);

  float max_dist_before_normalize = fmax(max_dist_hand, max_dist_obj);
  float buffer_dist = 1.03;

  max_dist_before_normalize *= buffer_dist;

  std::cout << "max dist hand new function " << max_dist_hand << std::endl;
  std::cout << "max dist object new function " << max_dist_obj << std::endl;

  SaveNormalizationParamsToNPZ(
    (-1 * center), (1.f / max_dist_before_normalize), normalizationOutputFile);

  float max_dist = NormalizationWithParams(geom, center, max_dist_before_normalize, true);
  float max_dist_2 = NormalizationWithParams(geomObj, center, max_dist_before_normalize, true);

  std::cout << "max dist " << max_dist << std::endl;

  std::vector<Eigen::Vector3f> xyz_hand, xyz_obj;
  std::vector<float> sdf_hand, sdf_obj;
  std::vector<int> point_labels_hand, point_labels_obj;


  std::vector<Eigen::Vector3f> temp_vert;
  std::vector<Eigen::Vector3f> vertices_hand, vertices_obj;
  std::vector<Eigen::Vector3f> normals_hand, normals_obj;
  std::vector<std::size_t> point_tri_ids_hand, point_tri_ids_obj;

  //For hand
  validate_point(geom, modelFaces, vis,
               max_dist,
               rejection_criteria_obs,
               rejection_criteria_tri,
               vertices_hand, normals_hand, point_tri_ids_hand,
               false);
  // build tree
  KdVertexList kdVerts_hand(vertices_hand);
  KdVertexListTree kdTree_surf_hand(3, kdVerts_hand);
  kdTree_surf_hand.buildIndex();

  // sample points hand
  sample_point(kdTree_surf_hand, geom, loaded_data,
               npyFileName, plyFileNameOut,
               num_sample,
               variance, second_variance,
               xyz_hand, sdf_hand, point_labels_hand,
               vertices_hand, normals_hand, point_tri_ids_hand, 
               save_ply, false);

  // For object 
  validate_point(geomObj, modelFacesObj, vis,
               max_dist,
               rejection_criteria_obs,
               rejection_criteria_tri,
               vertices_obj, normals_obj, point_tri_ids_obj,
               true);

  // build tree
  KdVertexList kdVerts_obj(vertices_obj);
  KdVertexListTree kdTree_surf_obj(3, kdVerts_obj);
  kdTree_surf_obj.buildIndex();
  // Sample points object
  sample_point(kdTree_surf_obj, geomObj , loaded_data,
               npyObjFileName, plyObjFileNameOut,
               num_sample,
               variance, second_variance,
               xyz_obj, sdf_obj, point_labels_obj,
               vertices_obj, normals_obj, point_tri_ids_obj, 
               save_ply, true);

  // build tree

  std::vector<float> sdf_hand_to_obj, sdf_obj_to_hand;
  std::vector<int> point_labels_hand_to_obj, point_labels_obj_to_hand;
  //// for calcalating distance to the other mesh
  calDistToOther(xyz_hand, kdTree_surf_obj, 
                 vertices_obj, normals_obj,
                 point_tri_ids_obj,
                 loaded_data,
                 sdf_hand_to_obj, point_labels_hand_to_obj,
                 variance,
                 true
                 );

  calDistToOther(xyz_obj, kdTree_surf_hand, 
                 vertices_hand, normals_hand,
                 point_tri_ids_hand,
                 loaded_data,
                 sdf_obj_to_hand, point_labels_obj_to_hand,
                 variance,
                 false
                 );

  writeSDFToNPZ(xyz_obj, sdf_obj, point_labels_obj, sdf_obj_to_hand, point_labels_obj_to_hand, npyObjFileName, true);

  writeSDFToNPZ(xyz_hand, sdf_hand, point_labels_hand, sdf_hand_to_obj, point_labels_hand_to_obj, npyFileName, true);

  return 0;
}
