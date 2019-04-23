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

import proguard.optimize.Optimizer;
import proguard.util.*;

import javax.swing.*;
import javax.swing.border.*;
import java.awt.*;
import java.awt.event.*;

/**
 * This <code>JDialog</code> allows the user to enter a String.
 *
 * @author Eric Lafortune
 */
final class OptimizationsDialog extends JDialog
{
    /**
     * Return value if the dialog is canceled (with the Cancel button or by
     * closing the dialog window).
     */
    public static final int CANCEL_OPTION = 1;

    /**
     * Return value if the dialog is approved (with the Ok button).
     */
    public static final int APPROVE_OPTION = 0;


    private final JCheckBox[] optimizationCheckBoxes = new JCheckBox[Optimizer.OPTIMIZATION_NAMES.length];

    private int returnValue;


    public OptimizationsDialog(JFrame owner)
    {
        super(owner, msg("selectOptimizations"), true);
        setResizable(true);

        // Create some constraints that can be reused.
        GridBagConstraints constraintsLast = new GridBagConstraints();
        constraintsLast.gridwidth = GridBagConstraints.REMAINDER;
        constraintsLast.anchor    = GridBagConstraints.WEST;
        constraintsLast.insets    = new Insets(1, 2, 1, 2);

        GridBagConstraints constraintsLastStretch = new GridBagConstraints();
        constraintsLastStretch.gridwidth = GridBagConstraints.REMAINDER;
        constraintsLastStretch.fill      = GridBagConstraints.HORIZONTAL;
        constraintsLastStretch.weightx   = 1.0;
        constraintsLastStretch.anchor    = GridBagConstraints.WEST;
        constraintsLastStretch.insets    = constraintsLast.insets;

        GridBagConstraints panelConstraints = new GridBagConstraints();
        panelConstraints.gridwidth = GridBagConstraints.REMAINDER;
        panelConstraints.fill      = GridBagConstraints.HORIZONTAL;
        panelConstraints.weightx   = 1.0;
        panelConstraints.weighty   = 0.0;
        panelConstraints.anchor    = GridBagConstraints.NORTHWEST;
        panelConstraints.insets    = constraintsLast.insets;

        GridBagConstraints selectButtonConstraints = new GridBagConstraints();
        selectButtonConstraints.weighty = 1.0;
        selectButtonConstraints.anchor  = GridBagConstraints.SOUTHWEST;
        selectButtonConstraints.insets  = new Insets(4, 4, 8, 4);

        GridBagConstraints okButtonConstraints = new GridBagConstraints();
        okButtonConstraints.weightx = 1.0;
        okButtonConstraints.weighty = 1.0;
        okButtonConstraints.anchor  = GridBagConstraints.SOUTHEAST;
        okButtonConstraints.insets  = selectButtonConstraints.insets;

        GridBagConstraints cancelButtonConstraints = new GridBagConstraints();
        cancelButtonConstraints.gridwidth = GridBagConstraints.REMAINDER;
        cancelButtonConstraints.weighty   = 1.0;
        cancelButtonConstraints.anchor    = GridBagConstraints.SOUTHEAST;
        cancelButtonConstraints.insets    = selectButtonConstraints.insets;

        GridBagLayout layout = new GridBagLayout();

        Border etchedBorder = BorderFactory.createEtchedBorder(EtchedBorder.RAISED);

        // Create the optimizations panel.
        JPanel optimizationsPanel     = new JPanel(layout);
        JPanel optimizationSubpanel   = null;
        String lastOptimizationPrefix = null;

        for (int index = 0; index < Optimizer.OPTIMIZATION_NAMES.length; index++)
        {
            String optimizationName = Optimizer.OPTIMIZATION_NAMES[index];

            String optimizationPrefix = optimizationName.substring(0, optimizationName.indexOf('/'));

            if (optimizationSubpanel == null || !optimizationPrefix.equals(lastOptimizationPrefix))
            {
                // Create a new keep subpanel and add it.
                optimizationSubpanel = new JPanel(layout);
                optimizationSubpanel.setBorder(BorderFactory.createTitledBorder(etchedBorder, msg(optimizationPrefix)));
                optimizationsPanel.add(optimizationSubpanel, panelConstraints);

                lastOptimizationPrefix = optimizationPrefix;
            }

            JCheckBox optimizationCheckBox = new JCheckBox(optimizationName);
            optimizationCheckBoxes[index] = optimizationCheckBox;

            optimizationSubpanel.add(tip(optimizationCheckBox, optimizationName.replace('/', '_')+"Tip"), constraintsLastStretch);
        }

        // Create the Select All button.
        JButton selectAllButton = new JButton(msg("selectAll"));
        selectAllButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                for (int index = 0; index < optimizationCheckBoxes.length; index++)
                {
                    optimizationCheckBoxes[index].setSelected(true);
                }
            }
        });

        // Create the Select All button.
        JButton selectNoneButton = new JButton(msg("selectNone"));
        selectNoneButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                for (int index = 0; index < optimizationCheckBoxes.length; index++)
                {
                    optimizationCheckBoxes[index].setSelected(false);
                }
            }
        });

        // Create the Ok button.
        JButton okButton = new JButton(msg("ok"));
        okButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                returnValue = APPROVE_OPTION;
                hide();
            }
        });

        // Create the Cancel button.
        JButton cancelButton = new JButton(msg("cancel"));
        cancelButton.addActionListener(new ActionListener()
        {
            public void actionPerformed(ActionEvent e)
            {
                hide();
            }
        });

        // Add all panels to the main panel.
        optimizationsPanel.add(selectAllButton,  selectButtonConstraints);
        optimizationsPanel.add(selectNoneButton, selectButtonConstraints);
        optimizationsPanel.add(okButton,         okButtonConstraints);
        optimizationsPanel.add(cancelButton,     cancelButtonConstraints);

        getContentPane().add(new JScrollPane(optimizationsPanel));
    }


    /**
     * Sets the initial optimization filter to be used by the dialog.
     */
    public void setFilter(String optimizations)
    {
        StringMatcher filter = optimizations != null && optimizations.length() > 0 ?
            new ListParser(new NameParser()).parse(optimizations) :
            new FixedStringMatcher("");

        for (int index = 0; index < Optimizer.OPTIMIZATION_NAMES.length; index++)
        {
            optimizationCheckBoxes[index].setSelected(filter.matches(Optimizer.OPTIMIZATION_NAMES[index]));
        }
    }


    /**
     * Returns the optimization filter composed from the settings in the dialog.
     */
    public String getFilter()
    {
        return new FilterBuilder(optimizationCheckBoxes, '/').buildFilter();
    }


    /**
     * Shows this dialog. This method only returns when the dialog is closed.
     *
     * @return <code>CANCEL_OPTION</code> or <code>APPROVE_OPTION</code>,
     *         depending on the choice of the user.
     */
    public int showDialog()
    {
        returnValue = CANCEL_OPTION;

        // Open the dialog in the right place, then wait for it to be closed,
        // one way or another.
        pack();
        setLocationRelativeTo(getOwner());
        show();

        return returnValue;
    }


    /**
     * Attaches the tool tip from the GUI resources that corresponds to the
     * given key, to the given component.
     */
    private static JComponent tip(JComponent component, String messageKey)
    {
        component.setToolTipText(msg(messageKey));

        return component;
    }


    /**
     * Returns the message from the GUI resources that corresponds to the given
     * key.
     */
    private static String msg(String messageKey)
    {
         return GUIResources.getMessage(messageKey);
    }
}