/*
 * ProGuard -- shrinking, optimization, obfuscation, and preverification
 *             of Java bytecode.
 *
 * Copyright (c) 2002-2019 Guardsquare NV
 *
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 2 of the License, or (at your option)
 * any later version.
 *
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, write to the Free Software Foundation, Inc.,
 * 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
 */
package proguard.gui.splash;

import java.awt.*;
import java.awt.RenderingHints.Key;
import java.awt.font.*;
import java.awt.geom.AffineTransform;
import java.awt.image.*;
import java.awt.image.renderable.RenderableImage;
import java.text.AttributedCharacterIterator;
import java.util.Map;

/**
 * This Graphics2D allows to fix some basic settings (Color, Font, Paint, Stroke,
 * XORMode) of a delegate Graphics2D, overriding any subsequent attempts to
 * change those settings.
 *
 * @author Eric Lafortune
 * @noinspection deprecation
 */
final class OverrideGraphics2D extends Graphics2D
{
    private final Graphics2D graphics;

    private Color  overrideColor;
    private Font   overrideFont;
    private Paint  overridePaint;
    private Stroke overrideStroke;
    private Color  overrideXORMode;

    private Color  color;
    private Font   font;
    private Paint  paint;
    private Stroke stroke;


    /**
     * Creates a new OverrideGraphics2D.
     * @param graphics the delegate Graphics2D.
     */
    public OverrideGraphics2D(Graphics2D graphics)
    {
        this.graphics = graphics;
        this.color    = graphics.getColor();
        this.font     = graphics.getFont();
        this.paint    = graphics.getPaint();
        this.stroke   = graphics.getStroke();
    }


    /**
     * Fixes the Color of the Graphics2D.
     *
     * @param color the fixed Color, or <code>null</code> to undo the fixing.
     */
    public void setOverrideColor(Color color)
    {
        this.overrideColor = color;
        graphics.setColor(color != null ? color : this.color);
    }

    /**
     * Fixes the Font of the Graphics2D.
     *
     * @param font the fixed Font, or <code>null</code> to undo the fixing.
     */
    public void setOverrideFont(Font font)
    {
        this.overrideFont = font;
        graphics.setFont(font != null ? font : this.font);
    }

    /**
     * Fixes the Paint of the Graphics2D.
     *
     * @param paint the fixed Paint, or <code>null</code> to undo the fixing.
     */
    public void setOverridePaint(Paint paint)
    {
        this.overridePaint = paint;
        graphics.setPaint(paint != null ? paint : this.paint);
    }

    /**
     * Fixes the Stroke of the Graphics2D.
     *
     * @param stroke the fixed Stroke, or <code>null</code> to undo the fixing.
     */
    public void setOverrideStroke(Stroke stroke)
    {
        this.overrideStroke = stroke;
        graphics.setStroke(stroke != null ? stroke : this.stroke);
    }

    /**
     * Fixes the XORMode of the Graphics2D.
     *
     * @param color the fixed XORMode Color, or <code>null</code> to undo the fixing.
     */
    public void setOverrideXORMode(Color color)
    {
        this.overrideXORMode = color;
        if (color != null)
        {
            graphics.setXORMode(color);
        }
        else
        {
            graphics.setPaintMode();
        }
    }


    // Implementations for Graphics2D.

    public void setColor(Color color)
    {
        this.color = color;
        if (overrideColor == null)
        {
            graphics.setColor(color);
        }
    }

    public void setFont(Font font)
    {
        this.font = font;
        if (overrideFont == null)
        {
            graphics.setFont(font);
        }
    }

    public void setPaint(Paint paint)
    {
        this.paint = paint;
        if (overridePaint == null)
        {
            graphics.setPaint(paint);
        }
    }

    public void setStroke(Stroke stroke)
    {
        this.stroke = stroke;
        if (overrideStroke == null)
        {
            graphics.setStroke(stroke);
        }
    }

    public void setXORMode(Color color)
    {
        if (overrideXORMode == null)
        {
            graphics.setXORMode(color);
        }
    }

    public void setPaintMode()
    {
        if (overrideXORMode == null)
        {
            graphics.setPaintMode();
        }
    }


    public Color getColor()
    {
        return overrideColor != null ? color : graphics.getColor();
    }

    public Font getFont()
    {
        return overrideFont != null ? font : graphics.getFont();
    }

    public Paint getPaint()
    {
        return overridePaint != null ? paint : graphics.getPaint();
    }

    public Stroke getStroke()
    {
        return overrideStroke != null ? stroke : graphics.getStroke();
    }


    public Graphics create()
    {
        OverrideGraphics2D g = new OverrideGraphics2D((Graphics2D)graphics.create());
        g.setOverrideColor(overrideColor);
        g.setOverrideFont(overrideFont);
        g.setOverridePaint(overridePaint);
        g.setOverrideStroke(overrideStroke);

        return g;
    }

    public Graphics create(int x, int y, int width, int height)
    {
        OverrideGraphics2D g = new OverrideGraphics2D((Graphics2D)graphics.create(x, y, width, height));
        g.setOverrideColor(overrideColor);
        g.setOverrideFont(overrideFont);
        g.setOverridePaint(overridePaint);
        g.setOverrideStroke(overrideStroke);

        return g;
    }


    // Delegation for Graphics2D

    public void addRenderingHints(Map hints)
    {
        graphics.addRenderingHints(hints);
    }

    public void clearRect(int x, int y, int width, int height)
    {
        graphics.clearRect(x, y, width, height);
    }

    public void clip(Shape s)
    {
        graphics.clip(s);
    }

    public void clipRect(int x, int y, int width, int height)
    {
        graphics.clipRect(x, y, width, height);
    }

    public void copyArea(int x, int y, int width, int height, int dx, int dy)
    {
        graphics.copyArea(x, y, width, height, dx, dy);
    }

    public void dispose()
    {
        graphics.dispose();
    }

    public void draw(Shape s)
    {
        graphics.draw(s);
    }

    public void draw3DRect(int x, int y, int width, int height, boolean raised)
    {
        graphics.draw3DRect(x, y, width, height, raised);
    }

    public void drawArc(int x, int y, int width, int height, int startAngle, int arcAngle)
    {
        graphics.drawArc(x, y, width, height, startAngle, arcAngle);
    }

    public void drawBytes(byte[] data, int offset, int length, int x, int y)
    {
        graphics.drawBytes(data, offset, length, x, y);
    }

    public void drawChars(char[] data, int offset, int length, int x, int y)
    {
        graphics.drawChars(data, offset, length, x, y);
    }

    public void drawGlyphVector(GlyphVector g, float x, float y)
    {
        graphics.drawGlyphVector(g, x, y);
    }

    public boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, Color bgcolor, ImageObserver observer)
    {
        return graphics.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, bgcolor, observer);
    }

    public boolean drawImage(Image img, int dx1, int dy1, int dx2, int dy2, int sx1, int sy1, int sx2, int sy2, ImageObserver observer)
    {
        return graphics.drawImage(img, dx1, dy1, dx2, dy2, sx1, sy1, sx2, sy2, observer);
    }

    public boolean drawImage(Image img, int x, int y, int width, int height, Color bgcolor, ImageObserver observer)
    {
        return graphics.drawImage(img, x, y, width, height, bgcolor, observer);
    }

    public boolean drawImage(Image img, int x, int y, int width, int height, ImageObserver observer)
    {
        return graphics.drawImage(img, x, y, width, height, observer);
    }

    public boolean drawImage(Image img, int x, int y, Color bgcolor, ImageObserver observer)
    {
        return graphics.drawImage(img, x, y, bgcolor, observer);
    }

    public boolean drawImage(Image img, int x, int y, ImageObserver observer)
    {
        return graphics.drawImage(img, x, y, observer);
    }

    public boolean drawImage(Image img, AffineTransform xform, ImageObserver obs)
    {
        return graphics.drawImage(img, xform, obs);
    }

    public void drawImage(BufferedImage img, BufferedImageOp op, int x, int y)
    {
        graphics.drawImage(img, op, x, y);
    }

    public void drawLine(int x1, int y1, int x2, int y2)
    {
        graphics.drawLine(x1, y1, x2, y2);
    }

    public void drawOval(int x, int y, int width, int height)
    {
        graphics.drawOval(x, y, width, height);
    }

    public void drawPolygon(int[] xPoints, int[] yPoints, int nPoints)
    {
        graphics.drawPolygon(xPoints, yPoints, nPoints);
    }

    public void drawPolygon(Polygon p)
    {
        graphics.drawPolygon(p);
    }

    public void drawPolyline(int[] xPoints, int[] yPoints, int nPoints)
    {
        graphics.drawPolyline(xPoints, yPoints, nPoints);
    }

    public void drawRect(int x, int y, int width, int height)
    {
        graphics.drawRect(x, y, width, height);
    }

    public void drawRenderableImage(RenderableImage img, AffineTransform xform)
    {
        graphics.drawRenderableImage(img, xform);
    }

    public void drawRenderedImage(RenderedImage img, AffineTransform xform)
    {
        graphics.drawRenderedImage(img, xform);
    }

    public void drawRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight)
    {
        graphics.drawRoundRect(x, y, width, height, arcWidth, arcHeight);
    }

    public void drawString(String s, float x, float y)
    {
        graphics.drawString(s, x, y);
    }

    public void drawString(String str, int x, int y)
    {
        graphics.drawString(str, x, y);
    }

    public void drawString(AttributedCharacterIterator iterator, float x, float y)
    {
        graphics.drawString(iterator, x, y);
    }

    public void drawString(AttributedCharacterIterator iterator, int x, int y)
    {
        graphics.drawString(iterator, x, y);
    }

    public boolean equals(Object obj)
    {
        return graphics.equals(obj);
    }

    public void fill(Shape s)
    {
        graphics.fill(s);
    }

    public void fill3DRect(int x, int y, int width, int height, boolean raised)
    {
        graphics.fill3DRect(x, y, width, height, raised);
    }

    public void fillArc(int x, int y, int width, int height, int startAngle, int arcAngle)
    {
        graphics.fillArc(x, y, width, height, startAngle, arcAngle);
    }

    public void fillOval(int x, int y, int width, int height)
    {
        graphics.fillOval(x, y, width, height);
    }

    public void fillPolygon(int[] xPoints, int[] yPoints, int nPoints)
    {
        graphics.fillPolygon(xPoints, yPoints, nPoints);
    }

    public void fillPolygon(Polygon p)
    {
        graphics.fillPolygon(p);
    }

    public void fillRect(int x, int y, int width, int height)
    {
        graphics.fillRect(x, y, width, height);
    }

    public void fillRoundRect(int x, int y, int width, int height, int arcWidth, int arcHeight)
    {
        graphics.fillRoundRect(x, y, width, height, arcWidth, arcHeight);
    }

    public Color getBackground()
    {
        return graphics.getBackground();
    }

    public Shape getClip()
    {
        return graphics.getClip();
    }

    public Rectangle getClipBounds()
    {
        return graphics.getClipBounds();
    }

    public Rectangle getClipBounds(Rectangle r)
    {
        return graphics.getClipBounds(r);
    }

    public Rectangle getClipRect()
    {
        return graphics.getClipRect();
    }

    public Composite getComposite()
    {
        return graphics.getComposite();
    }

    public GraphicsConfiguration getDeviceConfiguration()
    {
        return graphics.getDeviceConfiguration();
    }

    public FontMetrics getFontMetrics()
    {
        return graphics.getFontMetrics();
    }

    public FontMetrics getFontMetrics(Font f)
    {
        return graphics.getFontMetrics(f);
    }

    public FontRenderContext getFontRenderContext()
    {
        return graphics.getFontRenderContext();
    }

    public Object getRenderingHint(Key hintKey)
    {
        return graphics.getRenderingHint(hintKey);
    }

    public RenderingHints getRenderingHints()
    {
        return graphics.getRenderingHints();
    }

    public AffineTransform getTransform()
    {
        return graphics.getTransform();
    }

    public int hashCode()
    {
        return graphics.hashCode();
    }

    public boolean hit(Rectangle rect, Shape s, boolean onStroke)
    {
        return graphics.hit(rect, s, onStroke);
    }

    public boolean hitClip(int x, int y, int width, int height)
    {
        return graphics.hitClip(x, y, width, height);
    }

    public void rotate(double theta)
    {
        graphics.rotate(theta);
    }

    public void rotate(double theta, double x, double y)
    {
        graphics.rotate(theta, x, y);
    }

    public void scale(double sx, double sy)
    {
        graphics.scale(sx, sy);
    }

    public void setBackground(Color color)
    {
        graphics.setBackground(color);
    }

    public void setClip(int x, int y, int width, int height)
    {
        graphics.setClip(x, y, width, height);
    }

    public void setClip(Shape clip)
    {
        graphics.setClip(clip);
    }

    public void setComposite(Composite comp)
    {
        graphics.setComposite(comp);
    }

    public void setRenderingHint(Key hintKey, Object hintValue)
    {
        graphics.setRenderingHint(hintKey, hintValue);
    }

    public void setRenderingHints(Map hints)
    {
        graphics.setRenderingHints(hints);
    }

    public void setTransform(AffineTransform Tx)
    {
        graphics.setTransform(Tx);
    }

    public void shear(double shx, double shy)
    {
        graphics.shear(shx, shy);
    }

    public String toString()
    {
        return graphics.toString();
    }

    public void transform(AffineTransform Tx)
    {
        graphics.transform(Tx);
    }

    public void translate(double tx, double ty)
    {
        graphics.translate(tx, ty);
    }

    public void translate(int x, int y)
    {
        graphics.translate(x, y);
    }
}
