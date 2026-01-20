#include <iostream>
#include <string>
#include <vector>

#include "display.h"

#define FONT_SIZE 18

Display::Display()
{
    this->m_screen_width = 0;
    this->m_screen_height = 0;
    this->m_window = nullptr;
    this->m_renderer = nullptr;
    this->m_main_font = nullptr;
}
Display::~Display()
{
    this->distroy_renderer();
}
bool Display::create_renderer(std::string title, int screen_width, int screen_height)
{
    this->distroy_renderer();
    this->m_screen_width = screen_width;
    this->m_screen_height = screen_height;
    this->m_view_x_offset = 0.0;
    this->m_view_y_offset = 0.0;

    // Creating window
    this->m_window = SDL_CreateWindow(title.c_str(), SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, screen_width, screen_height, 0);

    // Handling if it fails to create window
    if (m_window == nullptr)
    {
        std::cout << "Window could not be created! SDL_Error : " << SDL_GetError() << std::endl;
        this->distroy_renderer();
        return false;
    }
    // Creating renderer
    this->m_renderer = SDL_CreateRenderer(this->m_window, -1, SDL_RENDERER_ACCELERATED | SDL_RENDERER_PRESENTVSYNC);

    // Handling if it fails to create renderer
    if (this->m_renderer == nullptr)
    {
        std::cout << "Renderer could not be created! SDL_Error: " << SDL_GetError() << std::endl;
        this->distroy_renderer();
        return false;
    }

    // Creating Font
    this->m_main_font = TTF_OpenFont("Roboto-Regular.ttf", FONT_SIZE);

    // Handling if it fails to create font
    if (this->m_main_font == nullptr)
    {
        std::cout << "Fail to load font! SDL_ttf Error :" << TTF_GetError() << std::endl;
        this->distroy_renderer();
        return false;
    }
    // Sucessfully created SDL environment
    return true;
}
void Display::distroy_renderer()
{
    // Destroy renderer if it exists
    if (this->m_renderer != nullptr)
    {
        SDL_DestroyRenderer(this->m_renderer);
        this->m_renderer = nullptr;
    }

    // Destroy window if it exists
    if (this->m_window != nullptr)
    {
        SDL_DestroyWindow(this->m_window);
        this->m_window = nullptr;
    }

    // Destroy font if it exists
    if (this->m_main_font != nullptr)
    {
        TTF_CloseFont(this->m_main_font);
        this->m_main_font = nullptr;
    }
}
void Display::show_screen()
{
    if (this->m_renderer != nullptr)
    {
        SDL_RenderPresent(this->m_renderer);
    }
}
void Display::clear_screen()
{
    if (this->m_renderer != nullptr)
    {
        SDL_SetRenderDrawColor(this->m_renderer, 0x00, 0x00, 0x00, 0xFF);
        SDL_RenderClear(this->m_renderer);
    }
}

void Display::draw_text(const std::string text, const Vector2 pos, const double scale = 1, const SDL_Color color = {0, 0, 0}, bool centered = false)
{
    // Creating SDL Surface
    SDL_Surface *surface = TTF_RenderText_Blended(this->m_main_font, text.c_str(), color);

    // Handling error
    if (surface == nullptr)
    {
        std::cout << "Unable to render text surface! SDL_ttf Error : " << TTF_GetError() << std::endl;
        return;
    }

    // Creating SDL Texture
    SDL_Texture *texture = SDL_CreateTextureFromSurface(this->m_renderer, surface);

    // Handling error
    if (texture == nullptr)
    {
        SDL_FreeSurface(surface);
        std::cout << "Unable to render text from the rendered text! SDL Error : " << SDL_GetError() << std::endl;
        return;
    }

    int width = surface->w * scale;
    int height = surface->h * scale;

    int x = pos.x;
    int y = pos.y;

    if (centered)
    {
        x -= width / 2;
        y -= height / 2;
    }

    SDL_Rect render_quad = {x, y, width, height};
    SDL_RenderCopy(this->m_renderer, texture, nullptr, &render_quad);
    SDL_DestroyTexture(texture);
    SDL_FreeSurface(surface);
}

void Display::set_render_draw_color(uint8_t red, uint8_t green, uint8_t blue, uint8_t alpha = 0xFF)
{
    SDL_SetRenderDrawColor(this->m_renderer, red, green, blue, alpha);
}
void Display::set_view(double x_offset, double y_offset)
{
    this->m_view_x_offset = x_offset - this->m_view_height / 2.0;
    this->m_view_y_offset = y_offset - this->m_view_width / 2.0;
}
void Display::set_view(double width, double height, double x_offset, double y_offset)
{
    this->m_view_width = fabs(width);
    this->m_view_height = fabs(height);
    this->m_view_x_offset = x_offset - this->m_view_height / 2.0;
    this->m_view_y_offset = y_offset - this->m_view_width / 2.0;
}
void Display::draw_line(const Vector2 &start_pos, const Vector2 &end_pos)
{
    Vector2 point1 = transform_point(start_pos);
    Vector2 point2 = transform_point(end_pos);
    SDL_RenderDrawLine(this->m_renderer, point1.x, point1.y, point2.x, point2.y);
}
void Display::draw_lines(const std::vector<Vector2> &points)
{
    for (unsigned int i = 1; i < points.size(); ++i)
    {
        this->draw_line(points[i - 1], points[i]);
    }
}
void Display::draw_lines(const std::vector<std::vector<Vector2>> &dataset)
{
    for (const std::vector<Vector2> &points : dataset)
    {
        this->draw_lines(points);
    }
}
Vector2 Display::transform_point(const Vector2 &point)
{
    double dx = point.x - this->m_view_x_offset;
    double dy = point.y - this->m_view_y_offset;

    double y = this->m_screen_height - (dx / this->m_view_height) * this->m_screen_height;
    double x = (dy / this->m_view_width) * this->m_screen_width;

    return Vector2(x, y);
}

std::vector<Vector2> transform_points(const std::vector<Vector2> &points, const Vector2 &position, const double rotation)
{
    double cos_theta = cos(rotation);
    double sin_theta = sin(rotation);
    std::vector<Vector2> transformed_points;

    for (const Vector2 &point : points)
    {
        double x = point.x * cos_theta - point.y * sin_theta + position.x;
        double y = point.x * sin_theta - point.y * cos_theta + position.y;

        transformed_points.push_back(Vector2(x, y));
    }

    return transformed_points;
}

std::vector<std::vector<Vector2>> transform_points(const std::vector<std::vector<Vector2>> &dataset, const Vector2 &position, const double rotation)
{
    std::vector<std::vector<Vector2>> transformed_dataset;

    for (const std::vector<Vector2> points : dataset)
    {
        transformed_dataset.push_back(transform_points(points, position, rotation));
    }
    return transformed_dataset;
}

std::vector<Vector2> offset_points(const std::vector<Vector2> &points, const Vector2 &offset)
{
    std::vector<Vector2> ofsetted_points;

    for (const Vector2 &point : points)
    {
        ofsetted_points.push_back(Vector2(point.x + offset.x, point.y + offset.y));
    }
    return ofsetted_points;
}
std::vector<std::vector<Vector2>> offset_points(const std::vector<std::vector<Vector2>> &dataset, const Vector2 &offset)
{
    std::vector<std::vector<Vector2>> ofsetted_datset;

    for (const std::vector<Vector2> points : dataset)
    {
        ofsetted_datset.push_back(offset_points(points, offset));
    }
    return ofsetted_datset;
}