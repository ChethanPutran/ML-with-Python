#ifndef INCLUDE_NL_KALMAN_DISPLAY_H
#define INCLUDE_NL_KALMAN_DISPLAY_H
#include <memory>
#include <string>
#include <iostream>
#include <SDL2/SDL.h>
#include <SDL2/SDL_ttf.h>
#include <vector>

template <typename... Args>
std::string string_format(const std::string &foramt, Args... args)
{
    size_t size = snprintf(nullptr, 0, foramt.c_str(), args...) + 1;

    if (size <= 0)
    {
        throw std::runtime_error("Error during formatting");
    }

    std::unique_ptr<char[]> buf(new char[size]);
    snprintf(buf.get(), size, foramt.c_str(), args...);

    return std::string(buf.get(), buf.get() + size - 1);
}

struct Vector2
{
    double x, y;
    Vector2()
    {
        x = 0.0;
        y = 0.0;
    }
    Vector2(double x_, double y_)
    {
        x = x_;
        y = y_;
    }
};

// Declaring transform and offset functions
std::vector<Vector2> transform_points(const std::vector<Vector2> &points, const Vector2 &position, const double rotation);
std::vector<std::vector<Vector2>> transform_points(const std::vector<std::vector<Vector2>> &dataset, const Vector2 &position, const double rotation);
std::vector<Vector2> offset_points(const std::vector<Vector2> &points, const Vector2 &offset);
std::vector<std::vector<Vector2>> offset_points(const std::vector<std::vector<Vector2>> &dataset, const Vector2 &offset);

class Display
{
public:
    Display();
    ~Display();

    bool create_renderer(std::string title, int screen_width, int screen_height);
    void distroy_renderer();
    void show_screen();
    void clear_screen();
    const double get_screen_width()
    {
        return this->m_screen_width;
    }
    const double get_screen_height()
    {
        return this->m_screen_height;
    }
    const double get_screen_aspect_ratio()
    {
        return (this->get_screen_width()) / (this->get_screen_height());
    }
    void draw_text(const std::string text, const Vector2 pos, const double scale = 1, const SDL_Color color = {0, 0, 0}, bool centered = false);
    void set_render_draw_color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = 0xFF);
    void set_view(double x_offset, double y_offset);
    void set_view(double width, double height, double x_offset, double y_offset);
    void draw_line(const Vector2 &start_pos, const Vector2 &end_pos);
    void draw_lines(const std::vector<Vector2> &points);
    void draw_lines(const std::vector<std::vector<Vector2>> &dataset);

private:
    Vector2 transform_point(const Vector2 &point);
    int m_screen_width;
    int m_screen_height;
    double m_view_width;
    double m_view_height;
    double m_view_x_offset;
    double m_view_y_offset;
    SDL_Window *m_window;
    SDL_Renderer *m_renderer;
    TTF_Font *m_main_font;
};

#endif // INCLUDE_NL_KALMAN_DISPLAY_H