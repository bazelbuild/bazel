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
package proguard.gui;

import javax.swing.*;
import java.awt.*;
import java.awt.event.*;


/**
 * This <code>Jpanel</code> is similar to a <code>JTabbedPane</code>.
 * It uses buttons on the left-hand side to switch between panels.
 * An image can be added below these buttons.
 * Some methods are provided to switch between tabs.
 *
 * @author Eric Lafortune
 */
public class TabbedPane
     extends JPanel
{
    private final CardLayout  cardLayout  = new CardLayout();
    private final JPanel      cardPanel   = new JPanel(cardLayout);
    private final ButtonGroup buttonGroup = new ButtonGroup();


    /**
     * Creates a new TabbedPane.
     */
    public TabbedPane()
    {
        GridBagLayout layout = new GridBagLayout();
        setLayout(layout);

        GridBagConstraints cardConstraints = new GridBagConstraints();
        cardConstraints.gridx      = 1;
        cardConstraints.gridy      = 0;
        cardConstraints.gridheight = GridBagConstraints.REMAINDER;
        cardConstraints.fill       = GridBagConstraints.BOTH;
        cardConstraints.weightx    = 1.0;
        cardConstraints.weighty    = 1.0;
        cardConstraints.anchor     = GridBagConstraints.NORTHWEST;

        add(cardPanel, cardConstraints);
    }


    /**
     * Adds a component with a given title to the tabbed pane.
     *
     * @param title     the title that will be used in the tab button.
     * @param component the component that will be added as a tab.
     */
    public Component add(final String title, Component component)
    {
        GridBagConstraints buttonConstraints = new GridBagConstraints();
        buttonConstraints.gridx  = 0;
        buttonConstraints.fill   = GridBagConstraints.HORIZONTAL;
        buttonConstraints.anchor = GridBagConstraints.NORTHWEST;
        buttonConstraints.ipadx  = 10;
        buttonConstraints.ipady  = 4;

        JToggleButton button = new JToggleButton(title);

        // Let the button react on the mouse press, instead of waiting for the
        // mouse release.
        button.setModel(new JToggleButton.ToggleButtonModel()
        {
            public void setPressed(boolean b)
            {
                if ((isPressed() == b) || !isEnabled())
                {
                    return;
                }

                if (!b && isArmed())
                {
                    setSelected(!this.isSelected());
                }

                if (b)
                {
                    stateMask |= PRESSED;
                }
                else
                {
                    stateMask &= ~PRESSED;
                }

                fireStateChanged();

                if (isPressed())
                {
                    fireActionPerformed(new ActionEvent(this, ActionEvent.ACTION_PERFORMED, getActionCommand()));
                }
            }

        });

        // Switch to the tab on a button press.
        button.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                cardLayout.show(cardPanel, title);
            }
        });

        // Only one button can be selected at the same time.
        buttonGroup.add(button);

        // If this is the first tab, make sure its button is selected.
        if (cardPanel.getComponentCount() == 0)
        {
            button.setSelected(true);
        }

        // Add the button and its panel.
        add(button, buttonConstraints);
        cardPanel.add(title, component);

        return component;
    }


    /**
     * Adds an image below the tab buttons, after all tabs have been added.
     * The image will only be as visible as permitted by the available space.
     *
     * @param image the image.
     * @return the component containing the image.
     */
    public Component addImage(final Image image)
    {
        GridBagConstraints imageConstraints = new GridBagConstraints();
        imageConstraints.gridx   = 0;
        imageConstraints.weighty = 1.0;
        imageConstraints.fill    = GridBagConstraints.BOTH;
        imageConstraints.anchor  = GridBagConstraints.SOUTHWEST;

        JButton component = new JButton(new ImageIcon(image));
        component.setFocusPainted(false);
        component.setFocusable(false);
        component.setRequestFocusEnabled(false);
        component.setRolloverEnabled(false);
        component.setMargin(new Insets(0, 0, 0, 0));
        component.setHorizontalAlignment(JButton.LEFT);
        component.setVerticalAlignment(JButton.BOTTOM);
        component.setPreferredSize(new Dimension(0, 0));

        add(component, imageConstraints);

        return component;
    }


    /**
     * Selects the first tab.
     */
    public void first()
    {
        cardLayout.first(cardPanel);
        updateButtonSelection();
    }


    /**
     * Selects the last tab.
     */
    public void last()
    {
        cardLayout.last(cardPanel);
        updateButtonSelection();
    }


    /**
     * Selects the previous tab.
     */
    public void previous()
    {
        cardLayout.previous(cardPanel);
        updateButtonSelection();
    }


    /**
     * Selects the next tab.
     */
    public void next()
    {
        cardLayout.next(cardPanel);
        updateButtonSelection();
    }


    /**
     * Lets the button selection reflect the currently visible panel.
     */
    private void updateButtonSelection()
    {
        int count = cardPanel.getComponentCount();
        for (int index = 0 ; index < count ; index++) {
            Component card = cardPanel.getComponent(index);
            if (card.isShowing())
            {
                JToggleButton button = (JToggleButton)getComponent(index+1);
                button.setSelected(true);
            }
        }
    }
}
